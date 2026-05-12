"""Generate docs/experiments/experiment_overview.md from active configs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "experiments"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "experiments" / "experiment_overview.md"


FIELD_PATTERNS = {
    "experiment_name": re.compile(r"^experiment_name:\s*(.+)$"),
    "run_name": re.compile(r"^run_name:\s*(.+)$"),
    "model_name": re.compile(r"^\s+name:\s*(.+)$"),
    "cache_namespace": re.compile(r"^\s+cache_namespace:\s*(.+)$"),
    "run_dir": re.compile(r"^\s+run_dir:\s*(.+)$"),
}


def _clean(value: str) -> str:
    return value.strip().strip("'\"")


def _parse_config(path: Path) -> Dict[str, str]:
    result = {
        "config": path.relative_to(PROJECT_ROOT).as_posix(),
        "experiment_name": "",
        "run_name": "",
        "model_name": "",
        "cache_namespace": "",
        "run_dir": "",
        "validation": "",
    }
    validations: List[str] = []
    in_model = False

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("model:"):
            in_model = True
            continue
        if line and not line.startswith(" "):
            in_model = False

        for field, pattern in FIELD_PATTERNS.items():
            match = pattern.match(line)
            if not match:
                continue
            if field == "model_name" and not in_model:
                continue
            result[field] = _clean(match.group(1))

        validation_match = re.match(r"^\s+- name:\s*(.+)$", line)
        if validation_match:
            validations.append(_clean(validation_match.group(1)))

    result["validation"] = ", ".join(validations)
    return result


def iter_active_configs() -> Iterable[Path]:
    """Yield only active experiment configs, excluding configs/archive."""
    yield from sorted(CONFIG_ROOT.glob("*/*.yaml"))


def render_overview(configs: Iterable[Path]) -> str:
    rows = [_parse_config(path) for path in configs]
    lines = [
        "# Experiment Overview",
        "",
        "Active experiment configs live under `configs/experiments/` and are run from the project root.",
        "",
        "| Experiment | Run | Model | Validation | Cache | Output | Config |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {experiment_name} | {run_name} | {model_name} | {validation} | "
            "{cache_namespace} | `{run_dir}` | `{config}` |".format(**row)
        )
    lines.extend(
        [
            "",
            "Archived configs under `configs/archive/` are intentionally excluded from this table.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render_overview(iter_active_configs()), encoding="utf-8")
    print(OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix())


if __name__ == "__main__":
    main()
