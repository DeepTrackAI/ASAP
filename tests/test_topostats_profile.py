"""Offline tests for the Python 3.12 contour profile and YAML selection.

Set ASAP_TOPOSTATS_UPSTREAM_WHEEL to the downloaded official 2.4.2 wheel to
also run the source-preservation and wheel-integrity checks.
"""
from __future__ import annotations

import ast
import base64
import csv
from email.parser import BytesParser
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
import zipfile

from packaging.specifiers import SpecifierSet

SEGMENTATION = Path(__file__).resolve().parents[1]/'tutorials/dna-imaging/segmentation'
sys.path.insert(0, str(SEGMENTATION))
import setup_topostats as setup
import unet_segmentation_helpers as seg


class ProfileWheelTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('ASAP_TOPOSTATS_UPSTREAM_WHEEL'), 'Set upstream wheel path')
    def test_profile_changes_only_audited_source_and_has_valid_record(self):
        upstream = Path(os.environ['ASAP_TOPOSTATS_UPSTREAM_WHEEL'])
        with tempfile.TemporaryDirectory() as directory:
            wheel = setup.build_contour_wheel(upstream, Path(directory))
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            self.assertEqual(digest, hashlib.sha256(
                setup.build_contour_wheel(upstream, Path(directory)).read_bytes()).hexdigest())
            with zipfile.ZipFile(upstream) as original, zipfile.ZipFile(wheel) as patched:
                modified = []
                for name in original.namelist():
                    if name.startswith('topostats/') and original.read(name) != patched.read(name):
                        modified.append(name)
                self.assertEqual(set(modified), {
                    'topostats/tracing/ordered_tracing.py', 'topostats/measure/feret.py'})
                metadata = BytesParser().parsebytes(patched.read(
                    f'topostats-{setup.PROFILE_VERSION}.dist-info/METADATA'))
                self.assertEqual(metadata['Version'], setup.PROFILE_VERSION)
                self.assertIn('3.12.13', SpecifierSet(metadata['Requires-Python']))
                self.assertNotIn('topoly', metadata.get_all('Requires-Dist'))
                self.assertEqual(
                    {r for r in BytesParser().parsebytes(original.read(
                        'topostats-2.4.2.dist-info/METADATA')).get_all('Requires-Dist') if r != 'topoly'},
                    set(metadata.get_all('Requires-Dist')))
                records = csv.reader(io.StringIO(patched.read(
                    f'topostats-{setup.PROFILE_VERSION}.dist-info/RECORD').decode()))
                for name, encoded_hash, size in records:
                    if encoded_hash:
                        data = patched.read(name)
                        actual = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
                        self.assertEqual(encoded_hash, 'sha256='+actual, name)
                        self.assertEqual(int(size), len(data), name)
                self.assertTrue(any('/licenses/' in name for name in patched.namelist()))
                # Prove the tracing AST is identical except the audited classifier.
                def tracing_ast(data):
                    tree = ast.parse(data)
                    tree.body = [n for n in tree.body if not (
                        isinstance(n, ast.ImportFrom) and n.module == 'topoly')]
                    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
                        if cls.name == 'OrderedTraceNodestats':
                            cls.body = [n for n in cls.body if not (
                                isinstance(n, ast.FunctionDef) and n.name == 'get_topology')]
                    return ast.dump(tree, include_attributes=False)
                name = 'topostats/tracing/ordered_tracing.py'
                self.assertEqual(tracing_ast(original.read(name)), tracing_ast(patched.read(name)))

    def test_rejects_unverified_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/'wrong.whl'
            file.write_bytes(b'not the pinned upstream release')
            with self.assertRaisesRegex(RuntimeError, 'unverified'):
                setup.build_contour_wheel(file, Path(directory))


class ConfigSelectionTests(unittest.TestCase):
    def setUp(self):
        self.previous = Path.cwd()
        os.chdir(SEGMENTATION)

    def tearDown(self):
        os.chdir(self.previous)

    def test_default_is_experiment_preset(self):
        selected = seg.select_topostats_yaml()
        self.assertEqual(selected, Path('configs/topostats_relaxed.yaml'))
        values = seg.exposed_topostats_settings(selected)
        self.assertEqual(values['confidence'], 0.3)
        self.assertEqual(values['grain_crop_padding'], 10)

    def test_custom_yaml_retains_nested_scientific_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            import yaml
            original = seg.load_topostats_config('configs/topostats_relaxed.yaml')
            original['filter']['gaussian_size'] = 2.25
            original['nodestats']['branch_pairing_length'] = 13.0
            custom = Path(directory)/'my_config.yaml'
            custom.write_text(yaml.safe_dump(original), encoding='utf-8')
            path = seg.select_topostats_yaml('custom', custom_yaml=custom)
            prepared = seg.build_topostats_config(path, {}, 'scan.spm', None, Path(directory)/'out')
            self.assertEqual(prepared['filter']['gaussian_size'], 2.25)
            self.assertEqual(prepared['nodestats']['branch_pairing_length'], 13.0)
            self.assertEqual(original, seg.load_topostats_config(custom))
            self.assertTrue(prepared['output_dir'].is_absolute())

    def test_custom_path_is_required_and_unknown_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'TOPOSTATS_CUSTOM_YAML'):
            seg.select_topostats_yaml('custom')
        with self.assertRaises(ValueError):
            seg.select_topostats_yaml('typo')
        with self.assertRaises(FileNotFoundError):
            seg.select_topostats_yaml('custom', custom_yaml='not_a_real_yaml.yaml')

    def test_invalid_custom_yaml_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            import yaml
            from schema import SchemaError
            config = seg.load_topostats_config('configs/topostats_relaxed.yaml')
            config['filter']['gaussian_size'] = 'this is not a number'
            file = Path(directory)/'invalid.yaml'
            file.write_text(yaml.safe_dump(config), encoding='utf-8')
            with self.assertRaises(SchemaError):
                seg.build_topostats_config(file, {}, 'scan.spm', None, Path(directory)/'out')


@unittest.skipUnless(seg.topostats_runtime_info()['profile'] == 'asap-contour', 'Install contour profile')
class RuntimeProfileTests(unittest.TestCase):
    def test_real_processing_import_does_not_import_topoly(self):
        from topostats.processing import process_scan
        from topostats.tracing.ordered_tracing import OrderedTraceNodestats
        self.assertTrue(callable(process_scan))
        self.assertNotIn('topoly', sys.modules)
        self.assertEqual(OrderedTraceNodestats.get_topology(None, [[], []]), [None, None])

    def test_feret_filter_does_not_promote_colab_import_warning(self):
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter('default')
            import topostats.measure.feret as feret
            importlib.reload(feret)
            warnings.warn('_PyDriveImportHook.find_spec() not found', ImportWarning)
            self.assertTrue(any(isinstance(w.message, ImportWarning) for w in emitted))
            with self.assertRaises(RuntimeWarning):
                warnings.warn_explicit('numerical issue', RuntimeWarning,
                                       filename='feret.py', lineno=1, module='topostats.measure.feret')


class NotebookStructureTests(unittest.TestCase):
    def test_cells_are_explained_and_config_defaults_to_preset(self):
        notebook = json.loads((SEGMENTATION/'DNA_simulation_and_segmentation.ipynb').read_text(encoding='utf-8'))
        code = []
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                self.assertEqual(notebook['cells'][index-1]['cell_type'], 'markdown')
                source = ''.join(cell['source'])
                code.append(source)
                # No standalone function definitions have crept back into the notebook.
                self.assertFalse(any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                                     for n in ast.parse(source).body))
        full = '\n'.join(code)
        self.assertIn('TOPOSTATS_CONFIG_SOURCE = "preset"', full)
        self.assertIn('input("Path to your full TopoStats YAML file: ")', full)
        self.assertNotIn('filtered_settings', full)
        self.assertNotIn('warnings.filterwarnings("ignore"', full)


if __name__ == '__main__':
    unittest.main()
