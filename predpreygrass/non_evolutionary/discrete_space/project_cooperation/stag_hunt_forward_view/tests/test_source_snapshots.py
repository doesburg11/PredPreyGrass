"""Archived evaluations must load their own copied source without the checkout."""
import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


COOPERATION = Path(__file__).resolve().parents[2]


class TestSourceSnapshots(unittest.TestCase):
    def test_snapshot_loaders(self):
        for name in ("stag_hunt_forward_view", "stag_hunt_reputation"):
            for script in (COOPERATION / name).glob("*.py"):
                tree = ast.parse(script.read_text())
                functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
                if "copy_module_snapshot" not in functions or "load_predpreygrass_modules" not in functions:
                    continue
                with self.subTest(module=name, script=script.name), tempfile.TemporaryDirectory() as temp:
                    snapshot = Path(temp) / "REPRODUCE_CODE"
                    namespace = {
                        "Path": Path, "shutil": shutil, "__file__": str(script),
                        "_SNAPSHOT_EXCLUDE_DIRS": {"runs", "ray_results", "__pycache__"},
                    }
                    function = ast.Module(body=[functions["copy_module_snapshot"]], type_ignores=[])
                    exec(compile(function, str(script), "exec"), namespace)
                    namespace["copy_module_snapshot"](snapshot)
                    program = f'''
import runpy
from pathlib import Path
from importlib import import_module
script = Path({str(snapshot)!r}) / "predpreygrass/non_evolutionary/discrete_space/project_cooperation/{name}/{script.name}"
namespace = runpy.run_path(str(script), run_name="snapshot_check")
loader = namespace["load_predpreygrass_modules"]
loader()
source = import_module(loader.__globals__["PredPreyGrass"].__module__)
assert Path(source.__file__).is_relative_to({str(snapshot)!r}), source.__file__
'''
                    result = subprocess.run(
                        [sys.executable, "-I", "-c", program], cwd=temp,
                        capture_output=True, text=True, timeout=60,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
