"""Unsloth DDP training only. No generation or validation runs in this process."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import time

from .config import RunConfig
from .io import fingerprint, read_jsonl, write_json


def precision_report(model):
    elements = Counter()
    for parameter in model.parameters():
        elements[str(parameter.dtype)] += parameter.numel()
    forced = getattr(model, "_unsloth_forced_float32", None)
    return {"requested_base_storage": "float16",
            "unsloth_forced_float32": bool(forced if forced is not None else
                                           os.environ.get("UNSLOTH_FORCE_FLOAT32") == "1"),
            "parameter_elements_by_dtype": dict(elements)}


def language_lora_targets(model, torch):
    names = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
             "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"}
    return sorted({name.rsplit(".", 1)[-1] for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in names
                   and not any(part in name.lower() for part in ("visual", "vision", "image_encoder"))})


def train(config):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != config.world_size:
        raise ValueError("Launch training using torchrun --nproc_per_node=2")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    cache = config.path / "compile_cache" / f"rank{local_rank}"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")

    # Unsloth must patch Transformers before any ML imports in this process.
    from unsloth import FastLanguageModel
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    from .tokenization import CompletionCollator, prompt_ids

    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        raise RuntimeError("Two CUDA GPUs must be visible to each DDP process")
    torch.cuda.set_device(local_rank)
    set_seed(config.seed)
    model, processor = FastLanguageModel.from_pretrained(
        model_name=config.model, revision=config.revision, max_seq_length=config.max_length,
        dtype=torch.float16, load_in_4bit=False, load_in_16bit=True, full_finetuning=False,
        device_map={"": local_rank},
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    initial_precision = precision_report(model)
    force_fp32 = initial_precision["unsloth_forced_float32"]
    if force_fp32:
        # Qwen3.5 Gated DeltaNet overflows under FP16 autocast on T4.
        os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
    targets = language_lora_targets(model, torch)
    if not targets:
        raise RuntimeError("No language projection modules were found for LoRA")
    model = FastLanguageModel.get_peft_model(
        model, r=config.rank, lora_alpha=config.alpha, lora_dropout=0, bias="none",
        target_modules=targets, use_gradient_checkpointing="unsloth", random_state=config.seed,
        finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
    )
    model.config.use_cache = False
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Only LoRA parameters may be trainable")
    if any(any(word in name.lower() for word in ("visual", "vision", "image_encoder")) for name in trainable):
        raise RuntimeError("Vision parameters must remain frozen")
    rows = read_jsonl(config.prepared / "train.tokens.jsonl")
    source = read_jsonl(config.prepared / "train.jsonl")
    # Check that Unsloth did not silently change the inference template/token IDs.
    for raw, encoded in zip(source[:3], rows[:3], strict=True):
        prefix = prompt_ids(tokenizer, raw["messages"])
        if encoded["input_ids"][:len(prefix)] != prefix:
            raise RuntimeError("Tokenizer/template differs between preparation and Unsloth")
    if len(rows) < config.world_size * config.micro_batch:
        raise ValueError("Not enough prepared examples for a distributed microbatch")
    snapshot = {name: parameter.detach().cpu().clone() for name, parameter in trainable.items()} if config.profile == "smoke" else {}
    steps, gradient_checks = [], []

    class Checks(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.started = time.monotonic()

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            if config.profile != "smoke":
                return
            gradients = [p.grad for p in trainable.values() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
                raise RuntimeError("Smoke: missing or nonfinite LoRA gradients")
            if not any(torch.count_nonzero(g).item() for g in gradients):
                raise RuntimeError("Smoke: every LoRA gradient is zero")
            gradient_checks.append(state.global_step + 1)

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            steps.append(time.monotonic() - self.started)

        def on_log(self, args, state, control, logs=None, **kwargs):
            for key in ("loss", "grad_norm"):
                if key in (logs or {}) and not math.isfinite(float(logs[key])):
                    raise RuntimeError(f"Nonfinite training {key}")

    arguments = TrainingArguments(
        output_dir=str(config.path / "checkpoints"),
        per_device_train_batch_size=config.micro_batch,
        gradient_accumulation_steps=config.gradient_accumulation,
        num_train_epochs=config.epochs,
        max_steps=config.smoke_steps if config.profile == "smoke" else -1,
        learning_rate=config.learning_rate, warmup_ratio=.03, lr_scheduler_type="cosine",
        optim="adamw_torch", weight_decay=.01, max_grad_norm=1.0,
        fp16=not force_fp32, bf16=False,
        eval_strategy="no", save_strategy="steps", save_steps=100, save_total_limit=2,
        logging_steps=1 if config.profile == "smoke" else 10, logging_nan_inf_filter=False,
        report_to="none", seed=config.seed, data_seed=config.seed,
        dataloader_num_workers=0, remove_unused_columns=False,
        ddp_find_unused_parameters=False, ddp_broadcast_buffers=False,
        average_tokens_across_devices=True,
    )
    trainer = Trainer(model=model, args=arguments, train_dataset=rows,
                      data_collator=CompletionCollator(tokenizer.pad_token_id), callbacks=[Checks()])
    if force_fp32 and trainer.accelerator.mixed_precision != "no":
        raise RuntimeError("Unsloth requires no global autocast on Qwen3.5/T4")
    print(json.dumps({"rank": local_rank, "precision": initial_precision,
                      "trainer_fp16": trainer.args.fp16, "lora_targets": targets,
                      "examples": len(rows), "effective_batch": 8}), flush=True)
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    changed = any(not torch.equal(snapshot[name], parameter.detach().cpu())
                  for name, parameter in trainable.items()) if snapshot else None
    if config.profile == "smoke" and (not changed or len(gradient_checks) != config.smoke_steps):
        raise RuntimeError("Smoke did not complete every update with finite gradients and changed weights")
    if not all(torch.isfinite(p).all().item() for p in trainable.values()):
        raise RuntimeError("Final adapter contains nonfinite weights")
    adapter = config.path / "adapter"
    trainer.save_model(str(adapter))
    trainer.accelerator.wait_for_everyone()
    rank_report = {"rank": local_rank, "gpu": torch.cuda.get_device_name(local_rank),
                   "capability": list(torch.cuda.get_device_capability(local_rank)),
                   "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "step_seconds": steps, "gradient_checked_steps": gradient_checks,
                   "weights_changed": changed, "precision": precision_report(trainer.model)}
    write_json(config.path / f"train_rank{local_rank}.json", rank_report)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(adapter)
        manifest = json.loads((config.prepared / "manifest.json").read_text())
        lineage = {"role": "planner", "model": config.model, "revision": config.revision,
                   "thinking": False, "prompt_sha256": manifest["prompt_sha256"],
                   "data_contract_sha256": manifest["contract_sha256"],
                   "config_sha256": fingerprint(json.loads((config.path / "config.json").read_text())),
                   "checkpoint_selection": "final; no validation during training",
                   "global_step": trainer.state.global_step, "world_size": world_size,
                   "micro_batch": config.micro_batch, "gradient_accumulation": config.gradient_accumulation,
                   "trainable_parameters": sum(p.numel() for p in trainable.values()),
                   "target_modules": targets, "initial_precision": initial_precision,
                   "trainer_fp16": trainer.args.fp16, "trainer_bf16": trainer.args.bf16,
                   "accelerator_mixed_precision": trainer.accelerator.mixed_precision}
        write_json(adapter / "lineage.json", lineage)
        write_json(config.path / "training_report.json", {"metrics": result.metrics,
                   "history": trainer.state.log_history, "lineage": lineage,
                   "smoke": config.profile == "smoke"})
    trainer.accelerator.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    train(RunConfig.load(parser.parse_args().config))
