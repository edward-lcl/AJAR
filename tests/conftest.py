"""Shared pytest configuration.

Putting scripts/ on sys.path lets tests import the runner and aggregator
helpers directly via `import run_qwen3_gsm8k_mi`. We also expose a small
`load_script_module` fixture: tests that prefer the importlib path use it
because it registers the module in sys.modules first (dataclasses look up
their owning module there during type resolution).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_script_module(name: str) -> ModuleType:
    """Load a top-level script as a module, registering it in sys.modules.

    Required for any script that defines a `@dataclass`, because the dataclass
    machinery resolves `ClassVar` and similar annotations by looking up
    `sys.modules[cls.__module__]`. Without registration that lookup returns
    None and crashes inside `dataclasses.py`.
    """
    if name in sys.modules:
        return sys.modules[name]
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def runner_module() -> ModuleType:
    return load_script_module("run_qwen3_gsm8k_mi")


@pytest.fixture(scope="session")
def aggregator_module() -> ModuleType:
    return load_script_module("build_task6_table")
