#!/usr/bin/env python3
"""Run pytest with pytest-testmon and optional cached metadata.

Default behavior:
- If a cached .testmondata is found, run pytest with test selection.
- If no cache is found, run a full test pass while recording testmon data.

Usage:
  python scripts/test_changes.py -- pytest -q -m "not slow"
  python scripts/test_changes.py --full -- pytest tests/ -v
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Py<3.11 fallback
    tomllib = None


def _find_cached_testmon(cache_dir: Path) -> Path | None:
    if not cache_dir.exists():
        return None

    direct = cache_dir / ".testmondata"
    if direct.exists():
        return direct

    try:
        revs = subprocess.run(
            ["git", "rev-list", "--first-parent", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except subprocess.CalledProcessError:
        revs = []

    for sha in revs:
        for candidate in (
            cache_dir / sha / ".testmondata",
            cache_dir / f"testmondata-{sha}" / ".testmondata",
        ):
            if candidate.exists():
                return candidate

    latest = cache_dir / "testmondata-latest" / ".testmondata"
    if latest.exists():
        return latest

    for candidate in cache_dir.rglob(".testmondata"):
        return candidate

    return None


def _ensure_local_testmon(cache_dir: Path) -> bool:
    local = Path(".testmondata")
    if local.exists():
        return True

    cached = _find_cached_testmon(cache_dir)
    if cached is None:
        return False

    shutil.copy2(cached, local)
    return True


def _strip_leading_double_dash(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def _has_cov_flag(args: list[str]) -> bool:
    for arg in args:
        if arg == "--cov":
            return True
        if arg.startswith("--cov="):
            return True
    return False


def _has_cov_config_flag(args: list[str]) -> bool:
    return any(arg == "--cov-config" or arg.startswith("--cov-config=") for arg in args)


def _extract_coverage_report_settings(pyproject_path: Path) -> dict[str, object]:
    if tomllib is None or not pyproject_path.exists():
        return {}
    try:
        data = tomllib.loads(pyproject_path.read_text())
    except Exception:
        return {}
    coverage = data.get("tool", {}).get("coverage", {})
    report = coverage.get("report", {})
    return report if isinstance(report, dict) else {}


def _write_cov_config_without_branch(pyproject_path: Path) -> Path:
    report_settings = _extract_coverage_report_settings(pyproject_path)
    exclude_lines = report_settings.get("exclude_lines") or []
    fail_under = report_settings.get("fail_under")

    lines = ["[run]", "branch = False", ""]
    if exclude_lines or fail_under is not None:
        lines.append("[report]")
        if fail_under is not None:
            lines.append(f"fail_under = {fail_under}")
        if exclude_lines:
            lines.append("exclude_lines =")
            lines.extend(f"    {line}" for line in exclude_lines)

    content = "\n".join(lines).rstrip() + "\n"
    temp_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".coveragerc", delete=False
    )
    temp_file.write(content)
    temp_file.flush()
    temp_file.close()
    return Path(temp_file.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pytest with pytest-testmon.")
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("TESTMON_CACHE_DIR", ".testmon_cache"),
        help="Directory containing cached .testmondata artifacts.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full tests while recording testmon metadata.",
    )
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    pytest_args = _strip_leading_double_dash(args.pytest_args)
    if not pytest_args:
        pytest_args = ["pytest"]

    cache_dir = Path(args.cache_dir)
    has_cache = _ensure_local_testmon(cache_dir)

    has_testmon_flag = any(
        flag in pytest_args for flag in ("--testmon", "--testmon-noselect")
    )
    disable_testmon = "-p" in pytest_args and "no:testmon" in pytest_args
    has_cov = any(
        arg == "--cov"
        or arg.startswith("--cov=")
        or arg.startswith("--cov-report")
        or arg.startswith("--cov-fail-under")
        for arg in pytest_args
    )
    if has_cov and not disable_testmon:
        print(
            "[test_changes] warning: pytest-cov detected; disabling testmon to avoid conflicts.",
            file=sys.stderr,
        )
        pytest_args.extend(["-p", "no:testmon"])
        disable_testmon = True

    if not has_testmon_flag and not disable_testmon:
        if args.full or not has_cache:
            pytest_args.append("--testmon-noselect")
        else:
            pytest_args.append("--testmon")

    uses_testmon = any(
        flag in pytest_args for flag in ("--testmon", "--testmon-noselect")
    )
    temp_cov_config: Path | None = None
    if (
        uses_testmon
        and _has_cov_flag(pytest_args)
        and not _has_cov_config_flag(pytest_args)
    ):
        temp_cov_config = _write_cov_config_without_branch(Path("pyproject.toml"))
        pytest_args.extend(["--cov-config", str(temp_cov_config)])

    print("[test_changes] cache_dir=", cache_dir, file=sys.stderr)
    print("[test_changes] has_cache=", has_cache, file=sys.stderr)
    print("[test_changes] args=", " ".join(pytest_args), file=sys.stderr)

    try:
        result = subprocess.run(pytest_args)
    finally:
        if temp_cov_config is not None:
            try:
                temp_cov_config.unlink()
            except FileNotFoundError:
                pass
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
