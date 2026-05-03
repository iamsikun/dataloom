"""YAML config loading + validation + run-directory provisioning.

The config schema is intentionally small (no jsonschema dep). The
`load_run_config` function returns a typed dataclass; `prepare_run_dir`
copies the config and an env.json snapshot into results/{run_id}/.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ParallelConfig:
    unit: str = "cell"           # "cell" | "replication"
    n_jobs: int = -1
    backend: str = "loky"


@dataclass
class ReplicationsConfig:
    default: int = 200
    overrides: list[dict[str, Any]] = field(default_factory=list)

    def for_n(self, n: int) -> int:
        """Return R for cell size n; overrides win if n falls in their range."""
        for o in self.overrides:
            n_min = o.get("n_min", -1)
            n_max = o.get("n_max", float("inf"))
            if n_min <= n <= n_max:
                return int(o["R"])
        return int(self.default)


@dataclass
class RunConfig:
    experiment_id: str
    profile: str
    schema_version: int
    seed_root: int
    n_grid: list[int]
    beta_grid: list[float]
    rho_grid: list[float]
    constants: dict[str, float]
    estimators: list[str]
    replications: ReplicationsConfig
    parallel: ParallelConfig
    output_root: Path
    estimator_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunConfig":
        # Pull estimator-specific config blocks (any top-level key matching an
        # estimator name in `estimators` is treated as that estimator's config).
        estimators = list(raw["estimators"])
        estimator_configs = {
            name: raw[name]
            for name in estimators
            if name in raw and isinstance(raw[name], dict)
        }
        return cls(
            experiment_id=str(raw["experiment_id"]),
            profile=str(raw.get("profile", "smoke")),
            schema_version=int(raw.get("schema_version", 1)),
            seed_root=int(raw["seed_root"]),
            n_grid=[int(x) for x in raw["n_grid"]],
            beta_grid=[float(x) for x in raw["beta_grid"]],
            rho_grid=[float(x) for x in raw["rho_grid"]],
            constants={k: float(v) for k, v in raw["constants"].items()},
            estimators=estimators,
            replications=ReplicationsConfig(**raw.get("replications", {})),
            parallel=ParallelConfig(**raw.get("parallel", {})),
            output_root=Path(raw.get("output", {}).get("results_root", "results/")),
            estimator_configs=estimator_configs,
            raw=raw,
        )


def load_run_config(path: str | Path) -> RunConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return RunConfig.from_dict(raw)


def _git_short_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        )
        return bool(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def make_run_id(experiment_id: str) -> str:
    sha = _git_short_sha()
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix = "__dirty" if _git_dirty() else ""
    return f"{experiment_id}__{sha}__{ts}{suffix}"


def prepare_run_dir(
    config: RunConfig,
    config_path: str | Path,
    run_id: str | None = None,
) -> Path:
    """Create results/{run_id}/ and write config.yaml + env.json into it."""
    run_id = run_id or make_run_id(config.experiment_id)
    run_dir = config.output_root / run_id
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "aggregated").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    shutil.copyfile(config_path, run_dir / "config.yaml")

    env = {
        "python": sys.version,
        "platform": sys.platform,
        "hostname": socket.gethostname(),
        "git_sha": _git_short_sha(),
        "git_dirty": _git_dirty(),
        "cwd": os.getcwd(),
        "run_id": run_id,
    }
    try:
        import importlib.metadata as md
        env["packages"] = {
            pkg: md.version(pkg)
            for pkg in ("numpy", "scipy", "pandas", "joblib",
                        "matplotlib", "pyyaml", "pyarrow")
            if _has_pkg(pkg)
        }
    except Exception:  # noqa: BLE001
        env["packages"] = {}
    with open(run_dir / "env.json", "w") as f:
        json.dump(env, f, indent=2)
    return run_dir


def _has_pkg(name: str) -> bool:
    try:
        import importlib.metadata as md
        md.version(name)
        return True
    except md.PackageNotFoundError:
        return False
