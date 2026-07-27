from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SHIM_ROOT = Path(__file__).resolve().parent / "shim"
SHIM_RUNNER = SHIM_ROOT / "run_pytest.py"


def test_strict_shim_never_falls_back_to_upstream(tmp_path: Path) -> None:
    """Even a real tinygrad earlier via cwd cannot satisfy a missing name."""
    fake_tgrad_source = tmp_path / "tgrad_source"
    fake_upstream = tmp_path / "upstream"
    upstream_package = fake_upstream / "tinygrad"
    fake_tgrad_source.mkdir()
    upstream_package.mkdir(parents=True)

    (fake_tgrad_source / "tgrad.py").write_text(
        "class Tensor:\n    pass\n",
        encoding="utf-8",
    )
    (upstream_package / "__init__.py").write_text(
        "class Tensor:\n    pass\n\nfallback_only = object()\n",
        encoding="utf-8",
    )
    (upstream_package / "dtype.py").write_text(
        "upstream_dtype = object()\n",
        encoding="utf-8",
    )
    probe = fake_upstream / "test_probe.py"
    probe.write_text(
        """
import importlib
import pathlib
import subprocess
import sys

import pytest
import tgrad
import tinygrad


def test_only_tgrad_is_visible():
    assert tinygrad.Tensor is tgrad.Tensor
    assert pathlib.Path(tinygrad.__file__).resolve().parent.name == "tinygrad"

    with pytest.raises(AttributeError, match="refusing to fall back"):
        getattr(tinygrad, "fallback_only")

    with pytest.raises(ImportError):
        exec("from tinygrad import fallback_only")

    # A same-named upstream module exists beside this test.  The strict
    # package path must still make the import fail instead of finding it.
    with pytest.raises(ModuleNotFoundError, match="refusing to fall back"):
        importlib.import_module("tinygrad.dtype")

    shim_tensor = importlib.import_module("tinygrad.tensor")
    assert shim_tensor.Tensor is tgrad.Tensor

    child_code = (
        "import importlib\\n"
        "import tgrad\\n"
        "import tinygrad\\n"
        "assert tinygrad.Tensor is tgrad.Tensor\\n"
        "try:\\n"
        "    importlib.import_module('tinygrad.dtype')\\n"
        "except ModuleNotFoundError:\\n"
        "    pass\\n"
        "else:\\n"
        "    raise AssertionError('child silently imported upstream tinygrad.dtype')\\n"
    )
    child = subprocess.run(
        [sys.executable, "-c", child_code],
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stdout + child.stderr
""".lstrip(),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SHIM_ROOT), str(fake_tgrad_source)])
    env["PYTHONSAFEPATH"] = "1"
    p = subprocess.run(
        [sys.executable, str(SHIM_RUNNER), str(probe), "-q", "--no-header",
         "-p", "no:cacheprovider"],
        cwd=fake_upstream,
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    assert "1 passed" in p.stdout
