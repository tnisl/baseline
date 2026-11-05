"""Timed startup stages and native traceback dumps for otherwise silent stalls."""
from contextlib import contextmanager
import faulthandler
import json
import os
from pathlib import Path
import sys
import time


def memory_snapshot():
    result = {}
    for filename, fields in (("/proc/self/status", {"VmRSS", "VmHWM", "Threads"}),
                             ("/proc/meminfo", {"MemTotal", "MemAvailable", "SwapFree"})):
        path = Path(filename)
        if path.exists():
            for line in path.read_text().splitlines():
                key, _, value = line.partition(":")
                if key in fields:
                    result[key] = value.strip()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        result["cuda_device"] = torch.cuda.current_device()
        result["cuda_allocated_gib"] = torch.cuda.memory_allocated() / 2**30
        result["cuda_reserved_gib"] = torch.cuda.memory_reserved() / 2**30
    return result


@contextmanager
def startup_stage(output, rank, name, timeout=120):
    directory = Path(output) / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    event = {"rank": rank, "pid": os.getpid(), "stage": name}
    with (directory / f"rank{rank}.stages.jsonl").open("a", encoding="utf-8") as events, \
         (directory / f"rank{rank}.{name}.stacks.log").open("a", encoding="utf-8") as stacks:
        def emit(state):
            record = {**event, "event": state, "elapsed_s": time.monotonic()-started,
                      "memory": memory_snapshot()}
            line = json.dumps(record)
            events.write(line + "\n")
            events.flush()
            print(line, flush=True)
        emit("start")
        faulthandler.dump_traceback_later(timeout, repeat=True, file=stacks)
        try:
            yield
        except BaseException:
            emit("failed")
            raise
        else:
            emit("complete")
        finally:
            faulthandler.cancel_dump_traceback_later()
