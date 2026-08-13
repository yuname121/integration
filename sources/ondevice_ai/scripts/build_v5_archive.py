#!/usr/bin/env python3
"""Build and verify the standalone SafeNest V5 on-device AI ZIP."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import zipfile

ARCHIVE_NAME = "SafeNest_v5.0_ondevice_ai_package.zip"
ARCHIVE_ROOT = "SafeNest_V5_OnDevice_AI"
ROOT_FILES = (
    "README.md", "walkthrough.md", "requirements.txt",
    "requirements-pi.txt", "requirements-mac.txt",
)
DIRECTORIES = (
    "config", "inference", "sensors", "risk", "integrated_node",
    "models", "adapters", "benchmarks", "scripts", "tests", "docs",
)
REQUIRED_FILES = (
    "integrated_node/run_node.py", "integrated_node/runtime_config.py",
    "inference/inference_result.py", "risk/risk_engine.py",
    "risk/risk_rules.py", "risk/fallback.py", "sensors/base_sensor.py",
    "sensors/provider_contract.py", "sensors/thermal44/thermal44_driver.py",
    "sensors/mmwave/mmwave_adapter.py", "sensors/co2/co2_adapter.py",
    "sensors/pir/pir_adapter.py", "config/models.yaml", "config/sensors.yaml",
    "config/risk_rules.yaml", "models/model_manifest.json",
    "scripts/validate_v4_config.py", "scripts/build_v5_archive.py",
    "docs/TEAM_HANDOFF_GUIDE_V5.md", "docs/reports/V5_RELEASE_READINESS.md",
    "docs/reports/V5_SENSOR_PROVIDER_CONTRACT.md",
)
EXCLUDED_DIRS = {".venv", "venv", "__pycache__", ".pytest_cache", "releases"}
EXCLUDED_NAMES = {".DS_Store", "build_v4_archive.py"}
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".log", ".zip", ".tar", ".tar.gz", ".tgz")
MODEL_PATHS = (
    "models/thermal/thermal_fall_int8_v0.1.0.tflite",
    "models/mmwave/mmwave_resp_int8_v0.1.0.tflite",
    "models/co2/co2_occupancy_int8_v0.1.0.tflite",
)


class ArchiveBuildError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def excluded(path: Path) -> bool:
    if any(part in EXCLUDED_DIRS for part in path.parts[:-1]):
        return True
    if path.name in EXCLUDED_NAMES or path.as_posix().lower().endswith(EXCLUDED_SUFFIXES):
        return True
    return bool(
        path.parts
        and path.parts[0] == "benchmarks"
        and (path.name.lower().startswith("tmp_") or "temporary" in path.name.lower())
    )


def collect_source_files(root: Path) -> list[Path]:
    files = []
    for name in ROOT_FILES:
        if not (root / name).is_file():
            raise ArchiveBuildError(f"Required root file missing: {name}")
        files.append(Path(name))
    for directory in DIRECTORIES:
        base = root / directory
        if not base.is_dir():
            raise ArchiveBuildError(f"Required directory missing: {directory}")
        for source in base.rglob("*"):
            if source.is_symlink():
                raise ArchiveBuildError(f"Symlink forbidden: {source}")
            if source.is_file():
                relative = source.relative_to(root)
                if not excluded(relative):
                    files.append(relative)
    files = sorted(set(files), key=lambda item: item.as_posix())
    missing = sorted(set(REQUIRED_FILES) - {item.as_posix() for item in files})
    if missing:
        raise ArchiveBuildError(f"Required package files missing: {missing}")
    return files


def validate_models(root: Path) -> dict:
    manifest = json.loads((root / "models/model_manifest.json").read_text())
    entries = {item.get("path"): item for item in manifest["models"].values()}
    verified = {}
    for relative in MODEL_PATHS:
        entry = entries.get(relative)
        if not entry or str(entry.get("version")) != "0.1.0":
            raise ArchiveBuildError(f"Model version/manifest mismatch: {relative}")
        digest = sha256_file(root / relative)
        if digest != entry.get("sha256"):
            raise ArchiveBuildError(f"Model SHA-256 mismatch: {relative}")
        verified[relative] = {"version": "0.1.0", "sha256": digest}
    return verified


def zip_info(name: str, mode=0o644):
    info = zipfile.ZipInfo(name, date_time=(2026, 8, 3, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (mode & 0xFFFF) << 16
    return info


def verify_archive(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        self_set = set(names)
        if len(names) != len(self_set):
            raise ArchiveBuildError("Duplicate ZIP members")
        unsafe = [
            name for name in names
            if not name.startswith(ARCHIVE_ROOT + "/")
            or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
        ]
        if unsafe:
            raise ArchiveBuildError(f"Unsafe ZIP paths: {unsafe}")
        missing = sorted({f"{ARCHIVE_ROOT}/{x}" for x in REQUIRED_FILES} - self_set)
        if missing:
            raise ArchiveBuildError(f"ZIP required files missing: {missing}")
        checksum_name = f"{ARCHIVE_ROOT}/SHA256SUMS.txt"
        if checksum_name not in self_set:
            raise ArchiveBuildError("SHA256SUMS.txt missing")
        for line in archive.read(checksum_name).decode().splitlines():
            digest, member = line.split("  ", 1)
            if member not in self_set or sha256_bytes(archive.read(member)) != digest:
                raise ArchiveBuildError(f"ZIP checksum mismatch: {member}")
        bad = archive.testzip()
        if bad:
            raise ArchiveBuildError(f"ZIP CRC failure: {bad}")


def build_archive(project_root=None, output_dir=None):
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parent.parent
    destination = Path(output_dir).resolve() if output_dir else root.parent / "releases"
    destination.mkdir(parents=True, exist_ok=True)
    sources = collect_source_files(root)
    members = {}
    modes = {}
    for relative in sources:
        name = f"{ARCHIVE_ROOT}/{relative.as_posix()}"
        source = root / relative
        members[name] = source.read_bytes()
        modes[name] = source.stat().st_mode & 0o777
    package = {
        "archive_root": ARCHIVE_ROOT,
        "package_date": "2026-08-03",
        "project_version": "5.0",
        "model_versions_unchanged": True,
        "models": validate_models(root),
        "source_file_count": len(sources),
        "excluded_directory_names": sorted(EXCLUDED_DIRS),
        "excluded_file_names": sorted(EXCLUDED_NAMES),
        "excluded_suffixes": list(EXCLUDED_SUFFIXES),
    }
    manifest_name = f"{ARCHIVE_ROOT}/PACKAGE_MANIFEST.json"
    members[manifest_name] = (json.dumps(package, indent=2, sort_keys=True) + "\n").encode()
    modes[manifest_name] = 0o644
    checksum_name = f"{ARCHIVE_ROOT}/SHA256SUMS.txt"
    members[checksum_name] = "".join(
        f"{sha256_bytes(data)}  {name}\n" for name, data in sorted(members.items())
    ).encode()
    modes[checksum_name] = 0o644
    archive_path = destination / ARCHIVE_NAME
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(members.items()):
            archive.writestr(zip_info(name, modes[name]), data)
    verify_archive(archive_path)
    sidecar = archive_path.with_suffix(archive_path.suffix + ".sha256")
    sidecar.write_text(f"{sha256_file(archive_path)}  {archive_path.name}\n")
    return archive_path, sidecar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    archive, sidecar = build_archive(output_dir=args.output_dir)
    print(archive)
    print(sidecar)
    print(sha256_file(archive))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
