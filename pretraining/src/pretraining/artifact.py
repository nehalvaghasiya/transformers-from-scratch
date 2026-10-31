"""Portable, checksummed checkpoint bundles; publication remains an explicit action."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REQUIRED = ("model.pt", "config.json", "vocab.json", "run_info.json")
OPTIONAL = ("pool_head.pt", "initial.pt", "experiment.json", "README.md", "LICENSE")


def export_artifact(model: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    for name in REQUIRED:
        if not (model / name).is_file():
            raise FileNotFoundError(model / name)
    files = [model / name for name in (*REQUIRED, *OPTIONAL) if (model / name).is_file()]
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        manifest = Path(directory) / "artifact.json"
        manifest.write_text(json.dumps({"format": 1, "files": hashes}, indent=2) + "\n")
        with tarfile.open(output, "w:gz") as archive:
            for path in [*files, manifest]:
                archive.add(path, arcname=path.name)
    result: dict[str, object] = {
        "archive": output.name,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "files": hashes,
    }
    output.with_name(output.name + ".sha256").write_text(f"{result['sha256']}  {output.name}\n")
    return result


def fetch_artifact(url: str, sha256: str, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    if not url.startswith(("https://", "file://")):
        raise ValueError("Use an HTTPS artifact URL or a local file:// URL")
    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / "model.tar.gz"
        with urllib.request.urlopen(url) as response, archive.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != sha256:
            raise ValueError("Archive checksum mismatch")
        extracted = Path(directory) / "extracted"
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            if any(
                not member.isfile() or Path(member.name).name != member.name for member in members
            ):
                raise ValueError("Expected flat regular artifact files")
            bundle.extractall(extracted, filter="data")
        manifest = json.loads((extracted / "artifact.json").read_text())
        if manifest.get("format") != 1 or not set(REQUIRED) <= manifest["files"].keys():
            raise ValueError("Invalid artifact manifest")
        for name, expected in manifest["files"].items():
            if (
                Path(name).name != name
                or hashlib.sha256((extracted / name).read_bytes()).hexdigest() != expected
            ):
                raise ValueError("Artifact content checksum mismatch")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), output)
