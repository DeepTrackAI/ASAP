"""Regression checks for legacy Colab hooks with warnings treated as errors.

Run from the repository root with Python 3.10-3.12:
    python -m unittest discover -s tests -p test_colab_import_compat.py
"""
from __future__ import annotations

import importlib
import importlib.abc
from pathlib import Path
import sys
import types
import unittest
import warnings

SEGMENTATION = (
    Path(__file__).resolve().parents[1] / "tutorials" / "dna-imaging" / "segmentation"
)
sys.path.insert(0, str(SEGMENTATION))
import unet_segmentation_helpers as seg


class CredentialLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.credentials_preserved = True


class ColabImportCompatibilityTests(unittest.TestCase):
    def test_legacy_hook_imports_under_strict_warnings_and_keeps_loader(self):
        probe = "_dna_colab_credential_probe"
        loader = CredentialLoader()
        calls = []

        def find_module(self, fullname, path=None):
            calls.append((fullname, path))
            return loader if fullname == probe else None

        hook_type = type(
            "_PyDriveImportHook",
            (),
            {
                "__module__": "google.colab._import_hooks._pydrive",
                "find_module": find_module,
            },
        )
        hook = hook_type()
        sys.meta_path.insert(0, hook)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                # Python 3.12 removed the legacy fallback altogether.
                expected_error = ImportWarning if sys.version_info[:2] < (3, 12) else ModuleNotFoundError
                with self.assertRaises(expected_error):
                    importlib.import_module(probe)
                before = list(warnings.filters)
                self.assertEqual(seg.prepare_colab_import_hooks(), 1)
                self.assertEqual(seg.prepare_colab_import_hooks(), 0)
                module = importlib.import_module(probe)
                self.assertTrue(module.credentials_preserved)
                self.assertIs(module.__loader__, loader)
                self.assertIn((probe, None), calls)
                self.assertIsNone(hook.find_spec("_not_a_pydrive_module", None))
                target = types.ModuleType(probe)
                self.assertIs(hook.find_spec(probe, ["probe-path"], target).loader, loader)
                self.assertIn((probe, ["probe-path"]), calls)
                self.assertEqual(warnings.filters, before)
                with self.assertRaisesRegex(RuntimeWarning, "numerical problem"):
                    warnings.warn("numerical problem", RuntimeWarning)
        finally:
            sys.meta_path.remove(hook)
            sys.modules.pop(probe, None)

    def test_modern_and_unrelated_hooks_are_not_changed(self):
        def find_spec(self, fullname, path=None, target=None):
            return None

        modern_type = type(
            "_PyDrive2ImportHook",
            (),
            {
                "__module__": "google.colab._import_hooks._pydrive2",
                "find_spec": find_spec,
            },
        )
        unrelated_type = type(
            "_PyDriveImportHook",
            (),
            {"__module__": "another_package", "find_module": lambda *args: None},
        )
        modern, unrelated = modern_type(), unrelated_type()
        sys.meta_path[:0] = [modern, unrelated]
        try:
            self.assertEqual(seg.prepare_colab_import_hooks(), 0)
            self.assertNotIn("find_spec", modern.__dict__)
            self.assertFalse(hasattr(unrelated, "find_spec"))
        finally:
            sys.meta_path.remove(modern)
            sys.meta_path.remove(unrelated)

    def test_version_check_automatically_adapts_hook(self):
        hook_type = type(
            "_PyDriveImportHook",
            (),
            {
                "__module__": "google.colab._import_hooks._pydrive",
                "find_module": lambda *args: None,
            },
        )
        hook = hook_type()
        sys.meta_path.insert(0, hook)
        try:
            # Config preparation and U-Net installation both call this
            # entry point before importing TopoStats internals.
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                self.assertTrue(seg.check_topostats_version().startswith("2.4."))
            self.assertTrue(callable(hook.find_spec))
        finally:
            sys.meta_path.remove(hook)


if __name__ == "__main__":
    unittest.main()
