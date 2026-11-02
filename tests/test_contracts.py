import json
from pathlib import Path
import tempfile
import unittest
from umilife.core import *
from umilife.data import grouped_split,annotation_tasks,normalized_row,build
from umilife.evaluate import score,maximum_matches,token_f1
from umilife.train import encode

ROOT=Path(__file__).resolve().parents[2]/"train+public_test"
class FakeTokenizer:
    def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=False):
        text="".join(f"<{m['role']}>{m['content']}</end>" for m in messages)
        if add_generation_prompt:text+="<assistant>"
        return list(text.encode()) if tokenize else text

class QueueBackend:
    def __init__(self,replies):self.replies=iter(replies);self.inputs=[]
    def generate(self,role,messages,remaining_s):
        self.inputs.append((role,messages))
        return dumps(next(self.replies))

class Contracts(unittest.TestCase):
    def test_completion_mask(self):
        t=FakeTokenizer();r={"messages":[{"role":"user","content":"secret context"}],"target":"target"}
        x=encode(t,r);prefix=t.apply_chat_template(r["messages"],add_generation_prompt=True)
        self.assertEqual(x["labels"][:len(prefix)],[-100]*len(prefix))
        self.assertEqual(bytes(v for v in x["labels"] if v!=-100).decode(),"target</end>")

    def test_split_groups_duplicate_requests(self):
        rows=[{"id":str(i),"audio_text":s} for i,s in enumerate(["Bật đèn!","bật đèn","tắt quạt","hủy lịch","xin chào"])]
        splits=grouped_split(rows);self.assertEqual(splits["0"],splits["1"])
        self.assertEqual(set(splits.values()),{"train","validation"})

    def test_scorer_duplicate_and_hash_gate(self):
        self.assertEqual(token_f1("xin chào", "hoàn toàn khác"),0)
        self.assertAlmostEqual(token_f1("chào bạn", "chào"),2/3)
        cat={"x":{"parameters":{"properties":{"content":{"x-match":"fuzzy"}}}}}
        call={"name":"x","arguments":{"content":"xin chào bạn"}}
        gold={"id":"1","tool_calls":[call],"state_hash":"right"}
        pred={"id":"1","tool_calls":[call,call],"state_hash":"right"}
        self.assertEqual(maximum_matches(pred["tool_calls"],gold["tool_calls"],cat),1)
        self.assertAlmostEqual(score(pred,gold,cat)["score"],2/3)
        pred["state_hash"]="wrong";self.assertEqual(score(pred,gold,cat)["score"],0)
        self.assertEqual(score({"id":"1","tool_calls":[],"state_hash":"a"},
                               {"id":"1","tool_calls":[],"state_hash":"a"},cat)["score"],1)

    @unittest.skipUnless(ROOT.exists(),"Competition data is local and not distributed")
    def test_training_annotation_cannot_see_current_or_future_result(self):
        row=read_jsonl(ROOT/"train/train.jsonl")[0];catalog=load_catalog(ROOT,"train")
        tasks=list(annotation_tasks([row],catalog,{row["id"]:"train"}))
        for i,task in enumerate(tasks):
            content=json.loads(task["messages"][-1]["content"])
            self.assertEqual(len(content["previous_steps"]),i)
            self.assertEqual(content["next_action"],row["label"]["tool_calls"][i])
            self.assertNotIn("label",content);self.assertNotIn("final_state",content)
            self.assertNotIn("tool_results",content)

    @unittest.skipUnless(ROOT.exists(),"Competition data is local and not distributed")
    def test_inference_gold_replay_and_prompt_isolation(self):
        row=read_jsonl(ROOT/"train/train.jsonl")[0];catalog=load_catalog(ROOT,"train")
        ToolEnv=import_toolenv(ROOT,"train");env=ToolEnv("train",row["id"],root=ROOT)
        replies=[]
        for action in row["label"]["tool_calls"]:
            replies += [{"thought":"Thực hiện bước tiếp theo.","next":"caller","tool":action["name"]},action]
        replies += [{"thought":"Hoàn tất.","next":"summarizer"},{"answer":"Đã nhận kết quả."}]
        backend=QueueBackend(replies)
        pred,trace=Agent(backend).run(normalized_row(row),row["audio_text"],allowed_specs(row,catalog),env)
        self.assertEqual(pred["tool_calls"],row["label"]["tool_calls"])
        self.assertEqual(trace["reason"],"complete")
        payload=json.loads(backend.inputs[0][1][-1]["content"])
        self.assertEqual(payload["trajectory"],[])
        self.assertNotIn("label",payload);self.assertNotIn("final_state",payload)
        self.assertNotIn("audio_text",dumps(payload))

    @unittest.skipUnless(ROOT.exists(),"Competition data is local and not distributed")
    def test_invalid_proposal_is_not_executed(self):
        row=read_jsonl(ROOT/"train/train.jsonl")[0];catalog=load_catalog(ROOT,"train")
        ToolEnv=import_toolenv(ROOT,"train");env=ToolEnv("train",row["id"],root=ROOT)
        backend=QueueBackend([{"thought":"Đọc dự báo.","next":"caller","tool":"weather.get_forecast"},
                              {"name":"nonexistent","arguments":{}},
                              {"name":"nonexistent","arguments":{}}])
        pred,trace=Agent(backend).run(normalized_row(row),row["audio_text"],allowed_specs(row,catalog),env)
        self.assertEqual(pred["tool_calls"],[]);self.assertEqual(trace["reason"],"format_error")
    @unittest.skipUnless(ROOT.exists(),"Competition data is local and not distributed")
    def test_preparation_reuses_train_ids_and_keeps_validation_separate(self):
        import random
        rows=read_jsonl(ROOT/"train/train.jsonl");assignments=grouped_split(rows)
        selected=[]
        for partition in ("train","validation"):
            candidates=[r for r in rows if assignments[r["id"]]==partition]
            random.Random(42).shuffle(candidates)
            selected+=candidates[:4 if partition=="train" else 2]
        tasks=list(annotation_tasks(selected,load_catalog(ROOT,"train"),assignments))
        with tempfile.TemporaryDirectory() as tmp:
            temp=Path(tmp);rationale=temp/"rationales.jsonl"
            # This fixture tests data plumbing only and is never used to fit the 4B model.
            rationale.write_text("\n".join(dumps({"key":t["key"],"thought":"TEST FIXTURE ONLY"}) for t in tasks)+"\n")
            manifest=build(ROOT,temp/"prepared",rationale,limit=4)
            self.assertEqual(manifest["excluded"],[])
            for role in ("global",*ROLES):
                train=read_jsonl(temp/"prepared"/f"{role}.train.jsonl")
                valid=read_jsonl(temp/"prepared"/f"{role}.validation.jsonl")
                self.assertTrue(train);self.assertTrue(valid)
                train_ids={r["id"] for r in train};valid_ids={r["id"] for r in valid}
                self.assertFalse(train_ids & valid_ids)
                self.assertTrue(all(assignments[x]=="train" for x in train_ids))
                self.assertTrue(all(assignments[x]=="validation" for x in valid_ids))
                for record in train+valid:
                    payload=json.loads(record["messages"][-1]["content"])
                    self.assertNotIn("label",payload)
                    self.assertNotIn("final_state",payload)

if __name__=="__main__":unittest.main()
