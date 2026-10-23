"""Build/install the pinned ASAP contour profile of TopoStats for Python 3.10-3.12.

This is a local modification of GPL-3.0-only TopoStats 2.4.2, not an official
TopoStats release. The verified PyPI wheel is rebuilt with three small changes:
Python 3.12 is allowed; unused Jones-polynomial classification and its eager
topoly import/dependency are removed; Feret's numerical warning filter is scoped
to its own RuntimeWarnings. All scan processing and trace geometry stay upstream.

Usage: python setup_topostats.py
       python setup_topostats.py --build-only --cache-dir /path/to/cache
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
from email import policy
from email.parser import BytesParser
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

UPSTREAM_VERSION = "2.4.2"
PROFILE_VERSION = "2.4.2+asap312.1"
UPSTREAM_SHA256 = "813840fad6ef740b78e46a92c50cd16f86d308fe7307c24d70b10907918ab03c"
PYPI_METADATA_URL = "https://pypi.org/pypi/topostats/2.4.2/json"
PROFILE_FILENAME = f"topostats-{PROFILE_VERSION}-py3-none-any.whl"


def download_upstream(cache_dir: Path) -> Path:
    """Download only the 244 kB release wheel and verify its fixed digest."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    wheel = cache_dir / f"topostats-{UPSTREAM_VERSION}-py3-none-any.whl"
    if not wheel.exists():
        with urllib.request.urlopen(PYPI_METADATA_URL, timeout=60) as response:
            metadata = json.load(response)
        entry = next(item for item in metadata["urls"] if item["filename"] == wheel.name)
        with urllib.request.urlopen(entry["url"], timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != UPSTREAM_SHA256:
            raise RuntimeError("Downloaded TopoStats wheel does not match the pinned SHA-256.")
        wheel.write_bytes(data)
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != UPSTREAM_SHA256:
        raise RuntimeError(f"Cached TopoStats wheel has an unexpected digest: {wheel}")
    return wheel


def patch_ordered_tracing(source: str) -> str:
    """Omit only knot/link classification; retain the upstream tracing code."""
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "OrderedTraceNodestats")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "get_topology")
    lines = source.splitlines(keepends=True)
    replacement = (
        "    def get_topology(self, nxyz: npt.NDArray) -> list:\n"
        '        """Jones-polynomial classification is omitted by the ASAP contour profile.\n\n'
        "        None means uncomputed, not an unknot or a failed contour trace.\n"
        "        Crossing resolution, coordinate ordering and circularity are unchanged.\n"
        '        """\n'
        "        return [None for _ in nxyz]\n"
    )
    lines[method.lineno - 1:method.end_lineno] = [replacement]
    source = "".join(lines)
    eager_import = "from topoly import jones, translate_code\n"
    if source.count(eager_import) != 1:
        raise RuntimeError("Upstream Topoly import changed; refusing an unchecked patch.")
    return source.replace(eager_import, "# ASAP contour profile: no eager knot-classifier import.\n")


def build_contour_wheel(upstream: Path, output_dir: Path) -> Path:
    """Rebuild a verified upstream wheel, including valid metadata and RECORD."""
    if hashlib.sha256(upstream.read_bytes()).hexdigest() != UPSTREAM_SHA256:
        raise RuntimeError("Refusing to patch an unverified TopoStats wheel.")
    old_info = f"topostats-{UPSTREAM_VERSION}.dist-info"
    new_info = f"topostats-{PROFILE_VERSION}.dist-info"
    with zipfile.ZipFile(upstream) as archive:
        files = {item.filename.replace(old_info + "/", new_info + "/", 1):
                 archive.read(item.filename) for item in archive.infolist()
                 if not item.is_dir() and not item.filename.endswith("/RECORD")}

    ordered_path = "topostats/tracing/ordered_tracing.py"
    feret_path = "topostats/measure/feret.py"
    originals = {name: hashlib.sha256(files[name]).hexdigest()
                 for name in (ordered_path, feret_path)}
    files[ordered_path] = patch_ordered_tracing(files[ordered_path].decode()).encode()
    feret = files[feret_path].decode()
    broad_filter = 'warnings.filterwarnings("error")'
    if feret.count(broad_filter) != 1:
        raise RuntimeError("Upstream Feret warning filter changed; refusing an unchecked patch.")
    files[feret_path] = feret.replace(
        broad_filter,
        'warnings.filterwarnings("error", category=RuntimeWarning, '
        'module=r"^topostats\\.measure\\.feret$")',
    ).encode()

    metadata_path = new_info + "/METADATA"
    metadata = BytesParser(policy=policy.compat32).parsebytes(files[metadata_path])
    metadata.replace_header("Version", PROFILE_VERSION)
    metadata.replace_header("Requires-Python", ">=3.10,<3.13")
    metadata.replace_header("Summary", str(metadata["Summary"]) + " (ASAP contour profile)")
    requirements = metadata.get_all("Requires-Dist", [])
    if requirements.count("topoly") != 1:
        raise RuntimeError("Unexpected upstream Topoly requirement.")
    del metadata["Requires-Dist"]
    for requirement in requirements:
        if requirement != "topoly":
            metadata["Requires-Dist"] = requirement
    files[metadata_path] = metadata.as_bytes(policy=policy.compat32)
    receipt = {
        "profile": "asap-contour", "version": PROFILE_VERSION,
        "upstream_version": UPSTREAM_VERSION, "upstream_sha256": UPSTREAM_SHA256,
        "python": ">=3.10,<3.13", "knot_classification": False,
        "topology_values": "None for uncomputed Jones-polynomial classification",
        "license": "GPL-3.0-only; upstream license files retained",
        "patched_files": {name: {"upstream_sha256": originals[name],
                                  "profile_sha256": hashlib.sha256(files[name]).hexdigest()}
                          for name in originals},
    }
    files["topostats/asap_profile.json"] = json.dumps(receipt, indent=2).encode() + b"\n"
    record_path = new_info + "/RECORD"
    record = io.StringIO(newline="")
    writer = csv.writer(record)
    for name, data in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow((name, "sha256=" + digest, len(data)))
    writer.writerow((record_path, "", ""))
    files[record_path] = record.getvalue().encode()
    output_dir.mkdir(parents=True, exist_ok=True)
    wheel = output_dir / PROFILE_FILENAME
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 18, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return wheel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".topostats-build"))
    parser.add_argument("--upstream-wheel", type=Path, help="Use a previously downloaded verified wheel")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        raise SystemExit("The ASAP contour profile supports Python 3.10, 3.11 and 3.12.")
    upstream = args.upstream_wheel or download_upstream(args.cache_dir)
    wheel = build_contour_wheel(upstream, args.cache_dir)
    print(f"Built {wheel.name} from verified TopoStats {UPSTREAM_VERSION}.", flush=True)
    print("ASAP contour profile: scan processing enabled; Jones-polynomial knot/link classification omitted.",
          flush=True)
    if not args.build_only:
        subprocess.run([sys.executable, "-m", "pip", "install", str(wheel.resolve())], check=True)
        print("TopoStats setup complete. If this changed already-imported numerical libraries, "
              "restart the session and rerun the setup cells.")


if __name__ == "__main__":
    main()
