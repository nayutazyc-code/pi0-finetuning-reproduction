"""No NormStats vectors or wrapped numeric continuation lines leave status reads."""
import ast
import importlib.util
from pathlib import Path
import re
import unittest

spec=importlib.util.spec_from_file_location('progress_status',Path(__file__).resolve().parents[1]/'scripts/status_v6_progress.py')
status=importlib.util.module_from_spec(spec);spec.loader.exec_module(status)
node=next(n for n in ast.parse(status.CODE).body if isinstance(n,ast.FunctionDef) and n.name=='safe_lines')
namespace={'re':re};exec(compile(ast.Module(body=[node],type_ignores=[]),'status_filter','exec'),namespace)


class RedactionTests(unittest.TestCase):
    def test_only_scalar_progress_and_errors_survive_multiline_norm_dump(self):
        lines='DataConfig(norm_stats=NormStats(mean=array([1.,2.]),\n  3., 4., q99=array([5.]),\n  6.,7.]))}, transforms=...)\nStep 25: loss=0.015, grad_norm=0.1\nValueError: bad dataset'
        self.assertEqual(namespace['safe_lines'](lines),['Step 25: loss=0.015, grad_norm=0.1','ValueError: bad dataset'])
        self.assertEqual(namespace['safe_lines']('Step 25: loss=array([private])'),[])


if __name__=='__main__':unittest.main()
