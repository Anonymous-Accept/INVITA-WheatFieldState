"""Audit the reviewer package for local identity and path leaks."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}

TEXT_EXTENSIONS = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".pdf",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
}

PRINTABLE_BYTES = re.compile(rb"[\t\r\n -~]{4,}")


def _patterns(extra_terms: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    slash = "/"
    backslash = "\\"
    patterns = [
        (
            "email_address",
            re.compile(
                r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
                re.IGNORECASE,
            ),
        ),
        (
            "private_ipv4",
            re.compile(
                r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
                r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|"
                r"192\.168\.\d{1,3}\.\d{1,3}|"
                r"100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3})\b"
            ),
        ),
        ("credential_word", re.compile("pass" + "word", re.IGNORECASE)),
        ("internal_label", re.compile("OFFICIAL" + " - " + "INTERNAL", re.IGNORECASE)),
        ("office_sensitivity_label", re.compile("MS" + "IP" + r"_Label", re.IGNORECASE)),
        ("unix_home_path", re.compile(re.escape(slash + "home" + slash))),
        ("unix_media_path", re.compile(re.escape(slash + "media" + slash))),
        ("unix_mount_path", re.compile(re.escape(slash + "mnt" + slash))),
        (
            "windows_user_path",
            re.compile(r"(^|[\s\"'=])" + r"[A-Za-z]:" + re.escape(backslash)),
        ),
        (
            "windows_user_path_alt",
            re.compile(r"(^|[\s\"'=])" + r"[A-Za-z]:" + re.escape(slash)),
        ),
    ]
    for index, term in enumerate(extra_terms, 1):
        patterns.append((f"external_denylist_{index}", re.compile(re.escape(term), re.IGNORECASE)))
    return patterns


def iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            files.append(path)
    return files


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS or path.name in {".gitignore"}


def iter_scan_lines(path: Path) -> list[tuple[int, str]]:
    if path.suffix.lower() == ".pdf":
        chunks = PRINTABLE_BYTES.findall(path.read_bytes())
        return [
            (index, line)
            for index, chunk in enumerate(chunks, 1)
            for line in chunk.decode("utf-8", errors="ignore").splitlines()
        ]
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")
    return list(enumerate(text.splitlines(), 1))


def _extra_terms(path: Path | None) -> list[str]:
    if path is None:
        env_path = os.environ.get("INVITA_ANONYMITY_DENYLIST")
        path = Path(env_path) if env_path else None
    if path is None:
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def scan(root: Path, denylist_file: Path | None = None) -> list[tuple[Path, int, str, str]]:
    findings: list[tuple[Path, int, str, str]] = []
    patterns = _patterns(_extra_terms(denylist_file))
    for path in iter_files(root):
        if not is_text_file(path):
            continue
        for line_no, line in iter_scan_lines(path):
            for label, pattern in patterns:
                if pattern.search(line):
                    findings.append((path, line_no, label, line.strip()))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", type=Path)
    parser.add_argument(
        "--denylist-file",
        type=Path,
        help="Optional local newline-delimited sensitive-term list. Do not commit it.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    findings = scan(root, denylist_file=args.denylist_file)
    if findings:
        print("Anonymity audit failed:")
        for path, line_no, label, line in findings:
            rel = path.relative_to(root)
            print(f"{rel}:{line_no}: {label}: {line[:180]}")
        return 1

    print("Anonymity audit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
