from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SHIM_ROOT = Path(__file__).resolve().parent / "shim"
SHIM_RUNNER = SHIM_ROOT / "run_pytest.py"


def _shim_modules() -> tuple[str, ...]:
    package = SHIM_ROOT / "tinygrad"
    modules = []
    for path in package.rglob("*.py"):
        if path.name == "__init__.py":
            if path.parent == package:
                continue
            relative = path.parent.relative_to(package)
        else:
            relative = path.relative_to(package).with_suffix("")
        modules.append("tinygrad." + ".".join(relative.parts))
    return tuple(sorted(modules))


def test_strict_shim_never_falls_back_to_upstream(tmp_path: Path) -> None:
    """Every shim module wins over same-named upstream decoys, including in children."""
    from scripts.parity.shim.run_pytest import EXPOSED_MODULES

    fake_tgrad_source = tmp_path / "tgrad_source"
    fake_upstream = tmp_path / "upstream"
    upstream_package = fake_upstream / "tinygrad"
    fake_tgrad_source.mkdir()
    upstream_package.mkdir(parents=True)

    (fake_tgrad_source / "tgrad.py").write_text(
        """
import os
import sys

class Tensor:
    pass

class TgradTypeError(TypeError):
    pass

class NotInLeanScope(RuntimeError):
    pass

_SUPPORTED_DTYPES = {"bf16", "f32"}

def _numel(shape):
    value = 1
    for dimension in shape:
        value *= dimension
    return value
""".lstrip(),
        encoding="utf-8",
    )
    (upstream_package / "__init__.py").write_text(
        "class Tensor:\n    pass\n\nfallback_only = object()\n",
        encoding="utf-8",
    )
    (upstream_package / "fallback_only_module.py").write_text(
        "upstream_only = object()\n",
        encoding="utf-8",
    )

    exposed_modules = _shim_modules()
    assert set(exposed_modules) == EXPOSED_MODULES
    for module_name in exposed_modules:
        relative = Path(*module_name.split(".")[1:])
        local_module = SHIM_ROOT / "tinygrad" / relative
        if local_module.is_dir():
            destination = upstream_package / relative / "__init__.py"
        else:
            destination = (upstream_package / relative).with_suffix(".py")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("upstream_only = object()\n", encoding="utf-8")

    probe = fake_upstream / "test_probe.py"
    probe.write_text(
        f"""
import importlib
import pathlib
import subprocess
import sys

import pytest
import tgrad
import tinygrad

EXPOSED_MODULES = {exposed_modules!r}


def test_only_tgrad_is_visible():
    assert tinygrad.Tensor is tgrad.Tensor
    assert pathlib.Path(tinygrad.__file__).resolve().parent.name == "tinygrad"

    with pytest.raises(AttributeError, match="refusing to fall back"):
        getattr(tinygrad, "fallback_only")

    with pytest.raises(ImportError):
        exec("from tinygrad import fallback_only")

    # Every exposed shim module has a same-named upstream decoy.  All module
    # files must still resolve beneath the shim package and none may reveal
    # the decoy-only attribute.
    shim_package = pathlib.Path(tinygrad.__file__).resolve().parent
    for module_name in EXPOSED_MODULES:
        module = importlib.import_module(module_name)
        assert pathlib.Path(module.__file__).resolve().is_relative_to(shim_package)
        with pytest.raises(AttributeError):
            getattr(module, "upstream_only")

    # Every explicitly importable unavailable value/type must fail on use.
    unavailable = importlib.import_module("tinygrad._unsupported")
    for module_name in EXPOSED_MODULES:
        module = importlib.import_module(module_name)
        for value in vars(module).values():
            if isinstance(value, unavailable.UnsupportedCapability):
                with pytest.raises(unavailable.TgradCapabilityError):
                    value()
            elif isinstance(value, unavailable._UnsupportedTypeMeta):
                with pytest.raises(unavailable.TgradCapabilityError):
                    value()
    with pytest.raises(unavailable.TgradCapabilityError):
        tinygrad.dtypes.int8.itemsize

    # A real but unexposed upstream module remains inaccessible.
    with pytest.raises(ModuleNotFoundError, match="refusing to fall back"):
        importlib.import_module("tinygrad.fallback_only_module")

    shim_tensor = importlib.import_module("tinygrad.tensor")
    assert shim_tensor.Tensor is tgrad.Tensor

    child_code = '''
import importlib
import pathlib
import tgrad
import tinygrad
assert tinygrad.Tensor is tgrad.Tensor
exposed_modules = {exposed_modules!r}
shim_package = pathlib.Path(tinygrad.__file__).resolve().parent
for module_name in exposed_modules:
    module = importlib.import_module(module_name)
    assert pathlib.Path(module.__file__).resolve().is_relative_to(shim_package)
    assert not hasattr(module, "upstream_only")
try:
    importlib.import_module("tinygrad.fallback_only_module")
except ModuleNotFoundError:
    pass
else:
    raise AssertionError("child silently imported an upstream tinygrad module")
'''
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
