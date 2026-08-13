#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
datasets/co2/raw_reader.py
Phase C-A1 — CO₂ Safe Raw Reader and Source-Row Contract.

Provides a deterministic, read-only, provenance-preserving reader for the real
UCI Occupancy Detection dataset archive:
  datasets/raw_archives/external_datasets/occupancy+detection.zip

Every returned observation preserves full raw provenance traceability:
- source archive path and SHA-256
- source member name and SHA-256
- physical source line number
- exported source row identifier
- original raw timestamp string
- naive local clock semantics (SOURCE_ACQUISITION_CLOCK, UNVERIFIED timezone)
- raw typed physical measurements (Temperature, Humidity, Light, CO2, HumidityRatio)
- binary Occupancy label (0 or 1)

Schema Profile:
- Header: 7 named fields ("date","Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy")
- Data Rows: 8 physical CSV fields (Field 0 = exported dataframe index, Field 1 = date, Fields 2..6 = features, Field 7 = Occupancy)
"""

from __future__ import annotations
import os
import hashlib
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple


EXPECTED_ARCHIVE_REL_PATH = "datasets/raw_archives/external_datasets/occupancy+detection.zip"
EXPECTED_ARCHIVE_SIZE = 335713
EXPECTED_ARCHIVE_SHA256 = "4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a"

EXPECTED_MEMBER_METADATA = {
    "datatest.txt": {
        "size": 200766,
        "sha256": "1b92c7c1b2838963464fa891a610cf3c5db4becb7189189b29b330107a584c7f",
        "rows": 2665,
        "occ_0": 1693,
        "occ_1": 972,
    },
    "datatest2.txt": {
        "size": 699664,
        "sha256": "d026d1bd5aeccd4aff4f3b3710d48e40613bd5fc370db7e61bbdcaa50d985095",
        "rows": 9752,
        "occ_0": 7703,
        "occ_1": 2049,
    },
    "datatraining.txt": {
        "size": 596674,
        "sha256": "b2c4d0ce2b9e4e453c476f7125ef31aeec2d1f5c7f5572d0e80de3df6521ab56",
        "rows": 8143,
        "occ_0": 6414,
        "occ_1": 1729,
    },
}

EXPECTED_HEADER_FIELDS = [
    "date",
    "Temperature",
    "Humidity",
    "Light",
    "CO2",
    "HumidityRatio",
    "Occupancy",
]


class RawReaderError(Exception):
    """Base exception for safe raw reader failures."""

    pass


class ArchiveNotFoundError(RawReaderError):
    """Raised when the raw archive file does not exist."""

    pass


class ArchiveIntegrityError(RawReaderError):
    """Raised when raw archive size or SHA-256 hash fails verification."""

    pass


class MemberIntegrityError(RawReaderError):
    """Raised when a zip member is missing or fails checksum/size checks."""

    pass


class SchemaValidationError(RawReaderError):
    """Raised when header or row field structure violates expected schema."""

    pass


class SourceRowParseError(RawReaderError):
    """Raised when row values cannot be parsed safely according to type contract."""

    pass


@dataclass(frozen=True)
class CO2SourceRowObservation:
    """Immutable record representing a single parsed raw observation with full provenance."""

    source_archive_path: str
    source_archive_sha256: str
    source_member_name: str
    source_member_sha256: str
    source_physical_line_number: int
    source_row_identifier: str
    source_timestamp_raw: str
    timestamp_reference: str
    source_timezone: str
    utc_conversion_claimed: bool
    temperature: float
    humidity: float
    light: float
    co2: float
    humidity_ratio: float
    occupancy: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_repo_root() -> Path:
    """Returns canonical repository root containing AGENTS.md."""
    root = Path(__file__).parent.parent.parent
    if (root / "AGENTS.md").exists():
        return root
    return Path(os.getcwd())


def compute_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class UCIOccupancyRawReader:
    """Deterministic read-only reader for UCI Occupancy Detection raw archive."""

    def __init__(self, repo_root: Optional[Path] = None, archive_rel_path: Optional[str] = None):
        self.repo_root = repo_root or get_repo_root()
        self.archive_rel_path = archive_rel_path or EXPECTED_ARCHIVE_REL_PATH
        self.archive_abs_path = self.repo_root / self.archive_rel_path

    def verify_archive(self) -> Tuple[int, str]:
        if not self.archive_abs_path.exists():
            raise ArchiveNotFoundError(f"Raw archive not found at: {self.archive_abs_path}")

        size = self.archive_abs_path.stat().st_size
        sha256 = compute_sha256_file(self.archive_abs_path)

        if size != EXPECTED_ARCHIVE_SIZE:
            raise ArchiveIntegrityError(
                f"Archive size mismatch for {self.archive_rel_path}: expected {EXPECTED_ARCHIVE_SIZE}, got {size}"
            )
        if sha256 != EXPECTED_ARCHIVE_SHA256:
            raise ArchiveIntegrityError(
                f"Archive SHA-256 mismatch for {self.archive_rel_path}: expected {EXPECTED_ARCHIVE_SHA256}, got {sha256}"
            )

        return size, sha256

    def iter_observations(
        self, target_member: Optional[str] = None
    ) -> Generator[CO2SourceRowObservation, None, None]:
        archive_size, archive_sha = self.verify_archive()

        members_to_read = (
            [target_member]
            if target_member
            else sorted(list(EXPECTED_MEMBER_METADATA.keys()))
        )

        with zipfile.ZipFile(self.archive_abs_path, "r") as z:
            for member_name in members_to_read:
                if member_name not in EXPECTED_MEMBER_METADATA:
                    raise MemberIntegrityError(f"Unexpected or unverified raw archive member: {member_name}")

                try:
                    raw_bytes = z.read(member_name)
                except KeyError:
                    raise MemberIntegrityError(f"Archive member missing from zip: {member_name}")

                m_sha = compute_sha256_bytes(raw_bytes)
                m_size = len(raw_bytes)

                exp_meta = EXPECTED_MEMBER_METADATA[member_name]
                if m_size != exp_meta["size"]:
                    raise MemberIntegrityError(
                        f"Member size mismatch for {member_name}: expected {exp_meta['size']}, got {m_size}"
                    )
                if m_sha != exp_meta["sha256"]:
                    raise MemberIntegrityError(
                        f"Member SHA-256 mismatch for {member_name}: expected {exp_meta['sha256']}, got {m_sha}"
                    )

                lines = raw_bytes.decode("utf-8", errors="replace").splitlines()
                if not lines:
                    raise SchemaValidationError(f"Empty member file: {member_name}")

                raw_header = lines[0]
                header_fields = [f.strip(' "') for f in raw_header.split(",")]

                if header_fields != EXPECTED_HEADER_FIELDS:
                    raise SchemaValidationError(
                        f"Header mismatch in {member_name}: expected {EXPECTED_HEADER_FIELDS}, got {header_fields}"
                    )

                line_num = 1  # header line is line 1
                data_row_count = 0

                for line in lines[1:]:
                    line_num += 1
                    if not line.strip():
                        continue

                    parts = line.split(",")
                    n_parts = len(parts)

                    if n_parts != 8:
                        raise SchemaValidationError(
                            f"Physical row width error in {member_name} at line {line_num}: expected 8 fields, got {n_parts} ({line})"
                        )

                    row_id = parts[0].strip(' "')
                    ts_str = parts[1].strip(' "')

                    if not row_id:
                        raise SourceRowParseError(f"Empty row identifier in {member_name} at line {line_num}")
                    if not ts_str:
                        raise SourceRowParseError(f"Empty timestamp string in {member_name} at line {line_num}")

                    try:
                        temp = float(parts[2])
                        hum = float(parts[3])
                        light = float(parts[4])
                        co2 = float(parts[5])
                        hum_ratio = float(parts[6])
                    except ValueError as e:
                        raise SourceRowParseError(
                            f"Numeric parse failure in {member_name} at line {line_num}: {e} ({line})"
                        )

                    lbl_str = parts[7].strip(' "')
                    if lbl_str in ["0", "0.0"]:
                        occ = 0
                    elif lbl_str in ["1", "1.0"]:
                        occ = 1
                    else:
                        raise SourceRowParseError(
                            f"Invalid Occupancy label '{lbl_str}' in {member_name} at line {line_num}"
                        )

                    data_row_count += 1
                    yield CO2SourceRowObservation(
                        source_archive_path=self.archive_rel_path,
                        source_archive_sha256=archive_sha,
                        source_member_name=member_name,
                        source_member_sha256=m_sha,
                        source_physical_line_number=line_num,
                        source_row_identifier=row_id,
                        source_timestamp_raw=ts_str,
                        timestamp_reference="SOURCE_ACQUISITION_CLOCK",
                        source_timezone="UNVERIFIED",
                        utc_conversion_claimed=False,
                        temperature=temp,
                        humidity=hum,
                        light=light,
                        co2=co2,
                        humidity_ratio=hum_ratio,
                        occupancy=occ,
                    )

                if data_row_count != exp_meta["rows"]:
                    raise SchemaValidationError(
                        f"Row count mismatch in {member_name}: expected {exp_meta['rows']}, parsed {data_row_count}"
                    )

    def read_all_observations(
        self, target_member: Optional[str] = None
    ) -> List[CO2SourceRowObservation]:
        obs = list(self.iter_observations(target_member=target_member))
        if target_member is None:
            expected_total = sum(m["rows"] for m in EXPECTED_MEMBER_METADATA.values())
            if len(obs) != expected_total:
                raise SchemaValidationError(
                    f"Total observations count mismatch: expected {expected_total}, got {len(obs)}"
                )
        return obs
