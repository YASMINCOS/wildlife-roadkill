"""
Carrega os scripts numerados (ex.: 01_prepare_dataset.py) como módulos:
nomes que começam com dígito não podem ser importados com `import` normal.
"""
import importlib.util
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def load_script(filename):
    spec = importlib.util.spec_from_file_location(f"script_{Path(filename).stem}", SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def prepare():
    return load_script("01_prepare_dataset.py")


@pytest.fixture(scope="session")
def evaluate():
    return load_script("04_evaluate.py")
