"""Provenance and output guards for independent baseline jobs."""
import json
from pathlib import Path

from .data import digest


def checkpoint_signature(directory):
    path = Path(directory).resolve()
    if not path.exists():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(repo_id=str(directory))).resolve()
    if not (path / "config.json").is_file():
        raise ValueError("Expected a local HF checkpoint: %s" % directory)
    files = sorted(set(path.glob("*.bin")) | set(path.glob("*.safetensors"))
                   | set(path.glob("*.json")))
    weights = [f for f in files if f.suffix in (".bin", ".safetensors")]
    if not weights:
        raise ValueError("No model weights in %s" % directory)
    return digest({"path": str(path), "files": [(f.name, f.stat().st_size,
                                                f.stat().st_mtime_ns) for f in files]})


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    temp.replace(path)


def new_output_dir(path, inputs=()):
    path = Path(path).resolve()
    for source in inputs:
        if source and path == Path(source).resolve():
            raise ValueError("Output would overwrite input: %s" % source)
    if path.exists() and any(path.iterdir()):
        raise ValueError("Output directory must be empty: %s" % path)
    path.mkdir(parents=True, exist_ok=True)
    return path
