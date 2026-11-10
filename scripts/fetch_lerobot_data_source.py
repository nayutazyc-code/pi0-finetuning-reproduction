"""Fetch the fixed data-only source snapshot with an archive integrity check."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import urllib.request

root = Path(__file__).resolve().parent.parent
manifest = json.loads((root / "requirements/lerobot-source.json").read_text())
commit = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
if manifest["commit"] != commit:
    raise ValueError("LeRobot commit differs from the openpi lock.")
target = root / manifest["local_path"]
if not target.resolve().is_relative_to((root / ".vendor").resolve()):
    raise ValueError("Source target must remain inside .vendor.")
if not target.exists():
    with urllib.request.urlopen(f"https://codeload.github.com/huggingface/lerobot/tar.gz/{commit}", timeout=120) as response:
        raw = response.read()
    if hashlib.sha256(raw).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("Archive SHA256 mismatch.")
    target.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts[1:]
            if not parts or ".." in parts or member.issym() or member.islnk():
                continue
            output = target.joinpath(*parts)
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(archive.extractfile(member).read())
for name, expected in manifest["source_file_sha256"].items():
    if hashlib.sha256((target / name).read_bytes()).hexdigest() != expected:
        raise ValueError(f"Fixed source content differs: {name}")
print(target)
