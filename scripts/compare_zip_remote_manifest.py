"""Compare a local zip manifest with an optional remote manifest.

The script is intentionally read-only. It never copies files from the remote
machine; it only compares relative paths so reviewers can see whether the zip
package used to seed this repository differs from a separately supplied task
directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path


def zip_manifest(path: Path, strip_prefix: str | None) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        names = set()
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if strip_prefix and name.startswith(strip_prefix):
                name = name[len(strip_prefix) :]
            names.add(name)
        return names


def remote_manifest(remote: str, remote_root: str) -> set[str]:
    command = [
        "ssh",
        remote,
        "find",
        remote_root,
        "-type",
        "f",
        "-printf",
        "%P\\n",
    ]
    proc = subprocess.run(command, check=True, text=True, capture_output=True)
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def file_manifest(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, type=Path, help="Local zip package.")
    parser.add_argument(
        "--strip-prefix",
        default="",
        help="Optional prefix to remove from zip paths before comparison.",
    )
    parser.add_argument("--remote", help="SSH target such as user@host.")
    parser.add_argument("--remote-root", help="Remote directory to inspect with find.")
    parser.add_argument(
        "--remote-list",
        type=Path,
        help="Optional newline-delimited remote manifest captured separately.",
    )
    args = parser.parse_args()

    local = zip_manifest(args.zip, args.strip_prefix)
    if args.remote_list:
        remote = file_manifest(args.remote_list)
    elif args.remote and args.remote_root:
        remote = remote_manifest(args.remote, args.remote_root)
    else:
        parser.error("provide --remote and --remote-root, or --remote-list")

    report = {
        "zip_only": sorted(local - remote),
        "remote_only": sorted(remote - local),
        "shared_count": len(local & remote),
        "zip_count": len(local),
        "remote_count": len(remote),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
