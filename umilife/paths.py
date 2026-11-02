"""Locate split manifests without traversing audio or application environments."""
from collections import deque
from pathlib import Path


def find_data_root(search_root):
    search_root=Path(search_root)
    queue=deque([(search_root,0)])
    archives=[]
    seen=set()
    while queue:
        parent,depth=queue.popleft()
        if parent in seen:
            continue
        seen.add(parent)
        for suffix in ("", "train+public_test", "trainpublic_test/train+public_test"):
            candidate=parent/suffix
            if ((candidate/"train/train.jsonl").is_file() and
                    (candidate/"public_test/public_test.jsonl").is_file()):
                return candidate,None
        archive=parent/"train+public_test.zip"
        if archive.is_file():
            archives.append(archive)
        if depth>=5:
            continue
        for child in sorted(parent.iterdir()):
            name=child.name
            if (name.startswith((".", "__MACOSX", "baseline")) or
                    name in {"audio", "train", "public_test", "private_test", "models", "outputs"} or
                    name.endswith("_environment")):
                continue
            if child.is_dir():
                queue.append((child,depth+1))
    if archives:
        return None,archives[0]
    raise FileNotFoundError("Attach the authorised LifeMate train+public_test dataset.")
