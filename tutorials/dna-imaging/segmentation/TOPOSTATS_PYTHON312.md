# TopoStats and Python 3.12.13: scope, changes and validation

Investigated and validated locally on 23 October 2026. The notebook uses a
modified, explicitly versioned **ASAP contour profile** of TopoStats 2.4.2:
`2.4.2+asap312.1`. This is not an official TopoStats Python 3.12 release and
does not provide Jones-polynomial knot/link classification.

## What the notebook actually uses

Simulation in Part 1 does not call TopoStats. Part 2 `.npy`/image inference
also uses the notebook's own PyTorch code. Only raw `.spm` analysis calls the
following TopoStats APIs, through `unet_segmentation_helpers.py`:

| Notebook operation | TopoStats API or data |
|---|---|
| Configuration validation | `validation.validate_config`, `DEFAULT_CONFIG_SCHEMA` |
| Plotting defaults | `config.update_plotting_config`, bundled `plotting_dictionary.yaml` |
| Scan loading and calibration | `io.LoadScans.get_data`, `LoadScans.load_spm` (AFMReader) |
| Full processing | `processing.process_scan` |
| PyTorch U-Net connection | `grains.keras.models.load_model`, `grains.predict_unet`, `unet_masking.predict_unet` are adapted in memory for `.pt` checkpoints |
| Results and plots | `TopoStats.image`, `full_mask_tensor`, `grain_crops`, ordered traces, molecule splines/contours and node counts |

`process_scan` dispatches to these upstream stages:

| Stage | Entry point and implementation | Needed here? |
|---|---|---|
| Filtering | `run_filters` → `Filters.filter_image` | Yes: flattened height map |
| Grain detection/refinement | `run_grains` → `Grains.find_grains`, `improve_grain_segmentation_unet` | Yes: candidate crops and U-Net masks |
| Grain statistics | `run_grainstats` → `GrainStats.calculate_stats`, including Feret calculations | Yes: retained pipeline statistics |
| Skeletonisation/pruning | `run_disordered_tracing` → `trace_image_disordered` | Yes |
| Crossing resolution | `run_nodestats` → `nodestats_image` | Yes: crossing nodes/branch pairing |
| Coordinate ordering | `run_ordered_tracing` → `ordered_tracing_image`, `OrderedTraceNodestats.compile_trace` / `trace` | Yes: connected molecular traces |
| Knot/link identity | `OrderedTraceNodestats.get_topology` → `topoly.translate_code`, `topoly.jones` | No: the displayed results do not use Jones classifications |
| Splines and contour lengths | `run_splining` → `splining_image` | Yes |
| Curvature | `run_curvature_stats` → `calculate_curvature_stats_image` | Retained; enabled by the existing presets |
| Saved analysis | `io.save_topostats_file`, `TopoStats.calculate_image_statistics` | Retained |

Crossing counts come from the resolved nodes (`len(crop.nodes)`), not from
`topoly`. Contour lengths come from the molecule's spline, not from its knot
label. Circularity is calculated by the trace code independently of Jones
classification. A circular DNA sample can still be traced as linear when
the image/preset does not yield a closed trace; the compatibility build does
not force circularity or contour length to match a known reference.

## Why a normal installation fails

The [official 2.4.2 package](https://pypi.org/project/topostats/2.4.2/) declares
`Requires-Python: >=3.10,<3.12` and an unconditional `topoly` dependency.
Its `ordered_tracing.py` imports `jones` and `translate_code` at module import
time, even when only ordinary scan processing is wanted. The upstream
[Python 3.12 import regression](https://github.com/ilbsm/topoly_tutorial/issues/4)
describes the expensive `topoly` polynomial dictionary import. Removing a
notebook version check or using a different YAML does not change this
packaging/import restriction.

There is also a separate Colab problem: `measure/feret.py` calls
`warnings.filterwarnings("error")` globally. That can turn Python's warning
about an older Colab PyDrive finder into an exception. This is independent
of the YAML and of the `topoly` import.

## Exact compatibility build

`setup_topostats.py` uses only the Python standard library to download the
approximately 244 kB official PyPI wheel. It verifies this fixed SHA-256:

```text
813840fad6ef740b78e46a92c50cd16f86d308fe7307c24d70b10907918ab03c
```

It then builds a local wheel with these changes:

1. Version is `2.4.2+asap312.1` and `Requires-Python` is `>=3.10,<3.13`.
2. Remove the eager `topoly` import and dependency. Replace only
   `OrderedTraceNodestats.get_topology` with an explicit uncomputed result
   (`None` for each molecule). The rest of the tracing source is retained.
   The upstream no-crossing tracer's simple geometric labels remain upstream;
   they should not be interpreted as a computed Jones-polynomial result.
3. Scope Feret's numerical warning-as-error rule to `RuntimeWarning` in
   `topostats.measure.feret`. Other warnings are not hidden or globally promoted.
4. Retain all other files/dependencies and upstream GPL-3.0-only license
   files. Add a profile receipt recording original and modified source hashes,
   and regenerate the wheel's metadata and `RECORD` hashes.

No pretend `topoly` module is inserted, no `--ignore-requires-python` is used,
and no Git clone is needed. TensorFlow/Keras remain dependencies because
TopoStats' masking module imports them, even though this notebook replaces
its prediction with PyTorch. Removing those imports would require a separate
adapter audit and is unnecessary for Python 3.12 compatibility.

The helper exposes `topostats_runtime_info()` and includes the profile in
each run result. The plotting cell announces that Jones classification is
omitted. Older Colab PyDrive hooks are adapted only when needed, preserving
their authentication loader.

To install:

```bash
python setup_topostats.py
```

Section 0 does this automatically in Colab/Kaggle. Use a fresh session;
restart once if installation changes numerical libraries already imported.
The profile is intentionally pinned: a future TopoStats release needs a new
audit and patch, not a relaxed version check. Use official TopoStats on a
supported Python version for analyses that require Jones knot/link identity.

## YAML selection

The experiment's `configs/topostats_{relaxed,supercoiled,minicircle}.yaml`
files are unchanged. Default notebook settings are:

```python
TOPOSTATS_CONFIG_SOURCE = "preset"
TOPOSTATS_PRESET = "relaxed"
TOPOSTATS_CUSTOM_YAML = ""
```

Preset mode exposes the same fine-tuned settings for optional editing.
For a custom full YAML, choose `"custom"` and supply a path; a blank path
prompts the user. The complete nested scientific configuration is loaded and
validated against TopoStats 2.4.2's schema. It is not filtered down to the
flat preset-override names. Runtime scan/output paths, the selected PyTorch
checkpoint, one worker and disabled TopoStats plot saving are controlled by
the notebook in both modes. Invalid custom scientific settings raise a schema
error rather than silently reverting to a preset.

## Validation evidence

Testing used an isolated **CPython 3.12.13 on Windows**, not a hosted Colab
Linux/GPU session. The existing `openmm-cuda` environment was not changed.

Installed test versions included NumPy 2.0.2, TensorFlow 2.20.0, Keras 3.15.1,
PyTorch 2.14.1 (CPU), SciPy 1.18.1, scikit-image 0.26.0, pandas 3.1.0,
Deeplay 0.1.4 and DeepTrack 2.0.1. These are an observed test environment,
not additional hard pins imposed on Colab's existing PyTorch installation.

- The standalone installer completed and `pip check` reported no broken
  requirements. `topoly` was neither installed nor imported.
- A separate dependency resolution for x86-64 Linux and Python 3.12.13
  succeeded, including `openmm[cuda12]`; this checks package resolution,
  not execution on Colab's Linux/GPU runtime.
- All 43 notebook cells completed in a real Python 3.12.13 Jupyter kernel
  in a review copy: non-MD mode, four generated samples, existing checkpoint,
  then real `.spm` analysis. This took about 50 seconds; it did not retrain
  the networks or rerun a large MD dataset.
- After the requested publication edits, the actual repository notebook
  completed all 44 cells with its unchanged MD defaults and existing
  checkpoint in Python 3.12.13. This run took about 159 seconds and saved
  five PNG figures plus the MD animation in the notebook. It exercised
  the `PeakMemTracker` class moved to `dna_simulation_helpers.py`, and ended
  with the fine-tuned relaxed-preset `.spm` analysis and the citation footer.
- All four MD/non-MD simple/residual checkpoints loaded and returned finite
  DNA/crossing predictions at native image size.
- A short CPU OpenMM 8.6.1 relaxation check returned four finite coordinate
  frames, confirming the MD integrator API works under Python 3.12.13.
  This short check is not a replacement for a calibrated full MD relaxation.
- The final SPM cell also ran with the MD Res-U-Net on three native scans,
  using the unchanged population presets:

| Reference molecule | Recorded contour (nm) | Python 3.12.13 contour (nm) | Crossing nodes |
|---|---:|---:|---:|
| Relaxed plasmid, scan `00090`, grain 0 | 771.02 | 771.024316 | 0 |
| Supercoiled plasmid, scan `00010`, grain 0 | 764.50 | 764.500000 | 3 |
| Minicircle, scan `00039`, grain 3 | 115.15 | 115.177841 | 0 |

The recorded contours are the prior results in
`../common-assets/example-images/real_examples.json`, not freshly rerun
Python 3.10 controls. The small minicircle difference (0.028 nm) is reported
explicitly; this is not evidence of bitwise equivalence across library versions.
The scans yielded 3, 1 and 7 traced molecules, respectively. Saved plots show
filtered images, U-Net masks and splined traces.

The regression suite checks legacy/modern Colab import hooks, default preset
selection, nested custom YAML preservation and validation, notebook cell
structure, deterministic wheel construction, source preservation, license and
`RECORD` integrity, absence of `topoly`, and the scoped Feret warning rule:

```bash
# Optional: enable the source/RECORD audit using the already-downloaded official wheel.
# Set ASAP_TOPOSTATS_UPSTREAM_WHEEL to its absolute path first.
python -m unittest discover -s tests -v
```

The suite is run from the repository root. With the upstream wheel available,
all 12 tests passed. The source-preservation check verifies that all upstream
package source files except the two documented modules are byte-identical,
and that the ordered-tracing AST is unchanged outside the omitted import and
classification method.

Before treating a hosted GPU run as verified, run section 0 and the notebook
in a fresh Colab Python 3.12.13 session after the reviewed files are published
or uploaded together. Installation still needs the normal scientific package
downloads if they are not already present.
