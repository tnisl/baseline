"""Kaggle SDK fallback for pushing notebooks, live logs and private artifacts."""
from __future__ import annotations
import argparse
from collections import deque
import json
import os
from pathlib import Path
import re
import signal


def redact(text):
    return re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?<redacted>", text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-env", help="Name of an existing environment variable; never a token value")
    commands=parser.add_subparsers(dest="command",required=True)
    status=commands.add_parser("status")
    status.add_argument("kernel")
    logs=commands.add_parser("logs")
    logs.add_argument("kernel")
    logs.add_argument("--seconds",type=int,default=20)
    logs.add_argument("--tail",type=int,default=60)
    logs.add_argument("--output",type=Path)
    push=commands.add_parser("push")
    push.add_argument("folder")
    push.add_argument("--timeout",default="43200")
    push.add_argument("--accelerator",default="NvidiaTeslaT4")
    output=commands.add_parser("output")
    output.add_argument("kernel")
    output.add_argument("--output",type=Path,required=True)
    output.add_argument("--pattern",default=r"(metrics\.json|lineage\.json|status\.json|run_config\.json|source\.json)$")
    args=parser.parse_args()
    if args.token_env:
        os.environ["KAGGLE_API_TOKEN"]=os.environ[args.token_env]
    # The standard SDK also supports its own configured credential file.
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi()
    api.authenticate()
    if args.command=="status":
        print(api.kernels_status(args.kernel))
    elif args.command=="push":
        print(api.kernels_push(args.folder,timeout=args.timeout,acc=args.accelerator))
    elif args.command=="output":
        paths,token=api.kernels_output(args.kernel,str(args.output),file_pattern=args.pattern,
                                      force=True,quiet=True)
        print(json.dumps({"files":paths,"next_page_token":bool(token)}))
    else:
        if args.seconds<=0:
            raise ValueError("--seconds must be positive")
        class CaptureFinished(Exception):
            pass
        def finish(*_):
            raise CaptureFinished()
        previous=signal.signal(signal.SIGALRM,finish)
        signal.alarm(args.seconds)
        chunks=deque(maxlen=args.tail)
        handle=None
        try:
            if args.output:
                args.output.parent.mkdir(parents=True,exist_ok=True)
                handle=args.output.open("w",encoding="utf-8")
            for event in api.kernels_logs_stream(args.kernel):
                data=redact(event.get("data",""))
                chunks.append(data)
                if handle:
                    handle.write(data)
                    handle.flush()
        except CaptureFinished:
            pass
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM,previous)
            if handle:
                handle.close()
        print("".join(chunks))


if __name__=="__main__":
    main()
