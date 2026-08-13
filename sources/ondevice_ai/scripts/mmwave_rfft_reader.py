#!/usr/bin/env python3
"""Safe, bounded structural reader for the Phase A1 mmWave rFFT payloads.

The source payload is a zlib stream containing a protocol-5 pickle.  This
module never asks Python's pickle runtime to construct objects.  Instead it
uses :mod:`pickletools` as a non-executing opcode tokenizer and interprets a
small allowlist sufficient for the observed ``[rFFTs, rBins]`` NumPy array
pair.  Globals are represented symbolically and are never imported or called.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import math
import pickletools
import re
import statistics
import sys
import zipfile
import zlib
from dataclasses import dataclass
from typing import Any, BinaryIO, Iterable

import numpy as np


DEFAULT_MAX_COMPRESSED_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_DECOMPRESSED_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_TEXT_BYTES = 1024 * 1024


class RFFTReaderError(Exception):
    """Base class for deterministic reader failures."""


class DecompressionError(RFFTReaderError):
    """The zlib stream is invalid, incomplete, oversized, or ambiguous."""


class PayloadFormatError(RFFTReaderError):
    """The decompressed payload is not a supported structural format."""


class UnsafeSerializationError(PayloadFormatError):
    """The pickle requires an opcode or global outside the strict allowlist."""


class TimestampParseError(RFFTReaderError):
    """A radar timestamp member cannot be parsed exactly."""


class ChirpConfigError(RFFTReaderError):
    """A chirp configuration member is malformed."""


@dataclass
class _GlobalRef:
    module: str
    name: str


@dataclass
class _DTypeSpec:
    code: str
    byteorder: str | None = None
    state: tuple[Any, ...] | None = None

    def numpy_dtype(self) -> np.dtype:
        allowed = {"c16": "complex128", "f8": "float64"}
        if self.code not in allowed:
            raise UnsafeSerializationError(f"dtype code is not allowed: {self.code!r}")
        prefix = self.byteorder if self.byteorder in ("<", ">") else ""
        dtype = np.dtype(prefix + self.code)
        if dtype.name != allowed[self.code]:
            raise UnsafeSerializationError(f"unexpected dtype normalization: {dtype}")
        return dtype


_MARK = object()


def identify_payload_format(payload: bytes) -> str:
    """Identify formats without instantiating serialized objects."""
    if payload.startswith(b"\x93NUMPY"):
        return "NUMPY_NPY"
    if len(payload) >= 2 and payload[0] == 0x80 and payload[1] <= 5:
        return f"PYTHON_PICKLE_PROTOCOL_{payload[1]}"
    if payload.startswith(b"PK\x03\x04"):
        return "ZIP_CONTAINER"
    if payload[:1] in (b"{", b"["):
        return "JSON_TEXT_CANDIDATE"
    return "UNKNOWN_BINARY"


def _is_valid_zlib_header(header: bytes) -> bool:
    if len(header) < 2:
        return False
    cmf, flg = header[0], header[1]
    return (cmf & 0x0F) == 8 and (cmf >> 4) <= 7 and ((cmf << 8) + flg) % 31 == 0


def bounded_zlib_decompress(
    source: BinaryIO,
    *,
    max_compressed_bytes: int = DEFAULT_MAX_COMPRESSED_BYTES,
    max_decompressed_bytes: int = DEFAULT_MAX_DECOMPRESSED_BYTES,
    chunk_size: int = 64 * 1024,
    allow_trailing_data: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    """Stream and bound a single zlib member.

    The function rejects invalid headers, truncation, output over the configured
    limit, and by default all trailing/concatenated data.
    """
    if max_compressed_bytes <= 0 or max_decompressed_bytes <= 0:
        raise ValueError("decompression limits must be positive")

    decoder = zlib.decompressobj(zlib.MAX_WBITS)
    output: list[bytes] = []
    compressed_count = 0
    decompressed_count = 0
    header = b""
    trailing = bytearray()

    while True:
        chunk = source.read(chunk_size)
        if not chunk:
            break
        compressed_count += len(chunk)
        if compressed_count > max_compressed_bytes:
            raise DecompressionError(
                f"compressed payload exceeds {max_compressed_bytes} bytes"
            )
        if len(header) < 2:
            header = (header + chunk)[:2]
            if len(header) == 2 and not _is_valid_zlib_header(header):
                raise DecompressionError(f"invalid zlib header: {header.hex()}")

        if decoder.eof:
            trailing.extend(chunk)
            continue

        remaining = max_decompressed_bytes - decompressed_count
        try:
            piece = decoder.decompress(chunk, remaining + 1)
        except zlib.error as exc:
            raise DecompressionError(f"zlib decode failed: {exc}") from exc
        output.append(piece)
        decompressed_count += len(piece)
        if decompressed_count > max_decompressed_bytes or decoder.unconsumed_tail:
            raise DecompressionError(
                f"decompressed payload exceeds {max_decompressed_bytes} bytes"
            )
        if decoder.unused_data:
            trailing.extend(decoder.unused_data)

    if len(header) < 2:
        raise DecompressionError("zlib payload is shorter than its header")
    if not decoder.eof:
        raise DecompressionError("truncated or incomplete zlib stream")

    remaining = max_decompressed_bytes - decompressed_count
    try:
        final = decoder.flush(remaining + 1)
    except zlib.error as exc:
        raise DecompressionError(f"zlib flush failed: {exc}") from exc
    output.append(final)
    decompressed_count += len(final)
    if decompressed_count > max_decompressed_bytes:
        raise DecompressionError(
            f"decompressed payload exceeds {max_decompressed_bytes} bytes"
        )

    concatenated = len(trailing) >= 2 and _is_valid_zlib_header(bytes(trailing[:2]))
    if trailing and not allow_trailing_data:
        kind = "concatenated zlib stream" if concatenated else "trailing data"
        raise DecompressionError(f"{kind} detected ({len(trailing)} bytes)")

    payload = b"".join(output)
    metadata = {
        "compressed_size_bytes": compressed_count,
        "decompressed_size_bytes": len(payload),
        "compression_ratio": round(len(payload) / compressed_count, 6)
        if compressed_count
        else None,
        "radar_header_signature": header.hex(),
        "zlib_decode_success": True,
        "zlib_eof": decoder.eof,
        "unused_trailing_bytes": len(trailing),
        "concatenated_stream_detected": concatenated,
    }
    return payload, metadata


def bounded_zlib_decompress_bytes(
    data: bytes,
    **kwargs: Any,
) -> tuple[bytes, dict[str, Any]]:
    """Convenience wrapper used by synthetic tests."""
    return bounded_zlib_decompress(io.BytesIO(data), **kwargs)


def _pop_marked(stack: list[Any]) -> list[Any]:
    values: list[Any] = []
    while stack:
        value = stack.pop()
        if value is _MARK:
            values.reverse()
            return values
        values.append(value)
    raise UnsafeSerializationError("pickle MARK stack underflow")


def _allowed_reduce(function: Any, args: Any) -> Any:
    if not isinstance(args, tuple):
        raise UnsafeSerializationError("REDUCE arguments are not a tuple")
    if not isinstance(function, _GlobalRef):
        raise UnsafeSerializationError("REDUCE target is not an allowed symbolic global")

    if (function.module, function.name) == ("numpy", "dtype"):
        if len(args) != 3 or not isinstance(args[0], str):
            raise UnsafeSerializationError("unexpected NumPy dtype constructor arguments")
        if args[1:] != (False, True):
            raise UnsafeSerializationError("unexpected NumPy dtype flags")
        return _DTypeSpec(args[0])

    if function.name == "_frombuffer" and function.module in {
        "numpy.core.numeric",
        "numpy._core.numeric",
    }:
        if len(args) != 4:
            raise UnsafeSerializationError("unexpected NumPy _frombuffer argument count")
        raw, dtype_spec, shape, order = args
        if not isinstance(raw, (bytes, bytearray)):
            raise UnsafeSerializationError("NumPy buffer is not primitive byte storage")
        if not isinstance(dtype_spec, _DTypeSpec):
            raise UnsafeSerializationError("NumPy buffer dtype is not a restricted dtype")
        if not isinstance(shape, tuple) or not shape or not all(
            isinstance(value, int) and value >= 0 for value in shape
        ):
            raise UnsafeSerializationError("NumPy buffer shape is invalid")
        if order != "C":
            raise UnsafeSerializationError("only C-order NumPy buffers are accepted")
        count = math.prod(shape)
        dtype = dtype_spec.numpy_dtype()
        if count * dtype.itemsize != len(raw):
            raise UnsafeSerializationError("NumPy buffer length does not match shape and dtype")
        return np.frombuffer(raw, dtype=dtype, count=count).reshape(shape, order="C")

    raise UnsafeSerializationError(
        f"pickle global is not allowed: {function.module}.{function.name}"
    )


def decode_restricted_numpy_pickle(payload: bytes) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Decode only the observed protocol-5 NumPy array pair without execution."""
    detected = identify_payload_format(payload)
    if detected != "PYTHON_PICKLE_PROTOCOL_5":
        raise PayloadFormatError(f"unsupported payload format: {detected}")

    allowed_opcodes = {
        "APPENDS",
        "BINGET",
        "BININT",
        "BININT1",
        "BININT2",
        "BUILD",
        "BYTEARRAY8",
        "EMPTY_LIST",
        "EMPTY_TUPLE",
        "FRAME",
        "LONG_BINGET",
        "MARK",
        "MEMOIZE",
        "NEWFALSE",
        "NEWTRUE",
        "NONE",
        "PROTO",
        "REDUCE",
        "SHORT_BINUNICODE",
        "STACK_GLOBAL",
        "STOP",
        "TUPLE",
        "TUPLE1",
        "TUPLE2",
        "TUPLE3",
    }
    allowed_globals = {
        ("numpy", "dtype"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.numeric", "_frombuffer"),
    }
    stack: list[Any] = []
    memo: dict[int, Any] = {}
    opcode_counts: dict[str, int] = {}
    stop_position: int | None = None

    try:
        operations: Iterable[tuple[Any, Any, int]] = pickletools.genops(payload)
        for opcode, argument, position in operations:
            name = opcode.name
            opcode_counts[name] = opcode_counts.get(name, 0) + 1
            if name not in allowed_opcodes:
                raise UnsafeSerializationError(f"pickle opcode is not allowed: {name}")

            if name == "PROTO":
                if argument != 5:
                    raise UnsafeSerializationError(f"pickle protocol is not allowed: {argument}")
            elif name == "FRAME":
                if not isinstance(argument, int) or argument < 0:
                    raise UnsafeSerializationError("invalid pickle frame length")
            elif name == "EMPTY_LIST":
                stack.append([])
            elif name == "EMPTY_TUPLE":
                stack.append(())
            elif name == "MARK":
                stack.append(_MARK)
            elif name == "MEMOIZE":
                if not stack:
                    raise UnsafeSerializationError("MEMOIZE stack underflow")
                memo[len(memo)] = stack[-1]
            elif name in {"BINGET", "LONG_BINGET"}:
                if argument not in memo:
                    raise UnsafeSerializationError(f"invalid pickle memo reference: {argument}")
                stack.append(memo[argument])
            elif name == "SHORT_BINUNICODE":
                stack.append(argument)
            elif name == "BYTEARRAY8":
                stack.append(argument)
            elif name == "NONE":
                stack.append(None)
            elif name == "NEWFALSE":
                stack.append(False)
            elif name == "NEWTRUE":
                stack.append(True)
            elif name in {"BININT", "BININT1", "BININT2"}:
                stack.append(argument)
            elif name == "STACK_GLOBAL":
                if len(stack) < 2:
                    raise UnsafeSerializationError("STACK_GLOBAL stack underflow")
                global_name = stack.pop()
                module_name = stack.pop()
                key = (module_name, global_name)
                if key not in allowed_globals:
                    raise UnsafeSerializationError(
                        f"pickle global is not allowed: {module_name}.{global_name}"
                    )
                stack.append(_GlobalRef(module_name, global_name))
            elif name in {"TUPLE1", "TUPLE2", "TUPLE3"}:
                size = int(name[-1])
                if len(stack) < size:
                    raise UnsafeSerializationError(f"{name} stack underflow")
                values = stack[-size:]
                del stack[-size:]
                stack.append(tuple(values))
            elif name == "TUPLE":
                stack.append(tuple(_pop_marked(stack)))
            elif name == "REDUCE":
                if len(stack) < 2:
                    raise UnsafeSerializationError("REDUCE stack underflow")
                args = stack.pop()
                function = stack.pop()
                stack.append(_allowed_reduce(function, args))
            elif name == "BUILD":
                if len(stack) < 2:
                    raise UnsafeSerializationError("BUILD stack underflow")
                state = stack.pop()
                instance = stack[-1]
                if not isinstance(instance, _DTypeSpec) or not isinstance(state, tuple):
                    raise UnsafeSerializationError("BUILD is allowed only for restricted dtype state")
                if len(state) != 8 or state[0] != 3 or state[1] != "<":
                    raise UnsafeSerializationError("unexpected NumPy dtype state")
                instance.byteorder = state[1]
                instance.state = state
            elif name == "APPENDS":
                values = _pop_marked(stack)
                if not stack or not isinstance(stack[-1], list):
                    raise UnsafeSerializationError("APPENDS target is not a list")
                stack[-1].extend(values)
            elif name == "STOP":
                stop_position = position
                break
    except (ValueError, UnicodeDecodeError) as exc:
        raise PayloadFormatError(f"malformed pickle opcode stream: {exc}") from exc

    if stop_position is None or stop_position + 1 != len(payload):
        raise UnsafeSerializationError("pickle STOP is missing or followed by trailing bytes")
    if len(stack) != 1 or not isinstance(stack[0], list) or len(stack[0]) != 2:
        raise UnsafeSerializationError("pickle root must be exactly [rFFTs, rBins]")
    tensor, range_bins = stack[0]
    if not isinstance(tensor, np.ndarray) or not isinstance(range_bins, np.ndarray):
        raise UnsafeSerializationError("pickle list members must both be restricted arrays")
    if tensor.ndim != 3 or tensor.dtype.name != "complex128":
        raise PayloadFormatError(
            f"rFFT tensor contract mismatch: shape={tensor.shape}, dtype={tensor.dtype}"
        )
    if range_bins.ndim != 1 or range_bins.dtype.name != "float64":
        raise PayloadFormatError(
            f"range-bin contract mismatch: shape={range_bins.shape}, dtype={range_bins.dtype}"
        )
    if tensor.shape[2] != range_bins.shape[0]:
        raise PayloadFormatError("rFFT range dimension does not match rBins length")

    metadata = {
        "payload_format": "PYTHON_PICKLE_PROTOCOL_5_NUMPY_ARRAY_PAIR",
        "pickle_protocol": 5,
        "safe_decode_method": "PICKLETOOLS_ALLOWLISTED_SYMBOLIC_VM",
        "arbitrary_object_execution": False,
        "root_structure": "LIST_OF_RFFT_TENSOR_AND_RANGE_BINS",
        "opcode_counts": dict(sorted(opcode_counts.items())),
        "global_allowlist": ["numpy.dtype", "numpy.core.numeric._frombuffer"],
    }
    return tensor, range_bins, metadata


def parse_radar_timestamps(raw: bytes) -> dict[str, Any]:
    """Parse an exact headerless ISO-8601 timestamp series and compute deltas."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TimestampParseError(f"timestamp member is not UTF-8: {exc}") from exc
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise TimestampParseError("timestamp member is empty")

    timestamps: list[tuple[dt.datetime, int]] = []
    failures: list[dict[str, Any]] = []
    for line_number, value in enumerate(lines, 1):
        match = re.fullmatch(
            r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})?",
            value,
        )
        try:
            if match is None:
                raise ValueError("not an ISO-8601 timestamp with at most nanosecond precision")
            base, fraction, timezone = match.groups()
            fraction = (fraction or "").ljust(9, "0")
            microseconds = fraction[:6]
            submicro_nanoseconds = int(fraction[6:] or "0")
            timezone = "+00:00" if timezone == "Z" else (timezone or "")
            parsed = dt.datetime.fromisoformat(
                base + (f".{microseconds}" if microseconds else "") + timezone
            )
            timestamps.append((parsed, submicro_nanoseconds))
        except (ValueError, OverflowError) as exc:
            failures.append({"line": line_number, "value": value, "error": str(exc)})
    if failures:
        raise TimestampParseError(f"{len(failures)} timestamp rows failed to parse")
    if len(timestamps) < 2:
        raise TimestampParseError("at least two timestamps are required")

    deltas = []
    for index in range(1, len(timestamps)):
        current_dt, current_extra_ns = timestamps[index]
        previous_dt, previous_extra_ns = timestamps[index - 1]
        delta = (current_dt - previous_dt).total_seconds()
        delta += (current_extra_ns - previous_extra_ns) / 1_000_000_000
        deltas.append(delta)
    median_delta = statistics.median(deltas)
    mean_delta = statistics.fmean(deltas)
    duplicate_count = sum(value == 0 for value in deltas)
    backward_count = sum(value < 0 for value in deltas)
    gap_threshold = 2.0 * median_delta if median_delta > 0 else None
    large_gap_count = (
        sum(value > gap_threshold + 1e-12 for value in deltas)
        if gap_threshold is not None
        else 0
    )
    return {
        "timestamp_count": len(timestamps),
        "first_timestamp": lines[0],
        "last_timestamp": lines[-1],
        "timestamp_format": "ISO8601_HEADERLESS_UTF8",
        "timestamp_median_dt_seconds": round(median_delta, 9),
        "timestamp_mean_dt_seconds": round(mean_delta, 9),
        "timestamp_min_dt_seconds": round(min(deltas), 9),
        "timestamp_max_dt_seconds": round(max(deltas), 9),
        "duplicate_timestamp_count": duplicate_count,
        "backward_timestamp_count": backward_count,
        "large_gap_count": large_gap_count,
        "large_gap_threshold_seconds": round(gap_threshold, 9)
        if gap_threshold is not None
        else None,
        "empirical_frame_rate_hz": round(1.0 / median_delta, 9)
        if median_delta > 0
        else None,
    }


_CHIRP_INTERPRETATIONS = {
    "START_FREQ": "configured chirp start frequency (Hz)",
    "IDLE": "configured inter-chirp idle time (s)",
    "ADC_START": "configured delay from chirp start to ADC sampling (s)",
    "RAMP": "configured chirp ramp duration (s)",
    "SLOPE": "configured chirp frequency slope (Hz/s)",
    "TX_START": "configured TX start offset (s)",
    "ADC_SAMPLES": "configured ADC samples per chirp",
    "SAMPLING_RATE": "configured ADC sampling rate (samples/s)",
    "RX_GAIN": "configured receiver gain (dB)",
    "LOOPS": "configured loop count per frame",
    "PERIODICITY": "configured frame period (ms)",
    "B": "configured sampled chirp bandwidth (Hz)",
    "R_BIN": "configured range-bin spacing (m)",
    "R_MAX": "configured maximum represented range (m)",
    "LAMBDA": "configured wavelength (m)",
    "TX_ANTENNAS": "configured active transmit antenna count",
    "RX_ANTENNAS": "configured receive antenna count",
}


def parse_chirp_config(raw: bytes) -> dict[str, Any]:
    """Parse and preserve chirp fields while limiting interpretation to evidence."""
    try:
        config = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChirpConfigError(f"chirp config is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(config, dict) or not config:
        raise ChirpConfigError("chirp config root must be a non-empty JSON object")

    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    a0_pairs = tuple(
        sorted(
            (key, float(value) if isinstance(value, (int, float)) else str(value))
            for key, value in config.items()
        )
    )
    a0_canonical = json.dumps(dict(a0_pairs), sort_keys=True).encode("utf-8")
    field_records = [
        {
            "original_key": key,
            "original_value": config[key],
            "interpreted_meaning": _CHIRP_INTERPRETATIONS.get(key, "not interpreted"),
            "evidence": "DIRECT_CHIRP_CONFIG",
        }
        for key in sorted(config)
    ]
    tx = config.get("TX_ANTENNAS")
    rx = config.get("RX_ANTENNAS")
    virtual_count = tx * rx if isinstance(tx, int) and isinstance(rx, int) else None
    period_ms = config.get("PERIODICITY")
    frame_period = period_ms * 1e-3 if isinstance(period_ms, (int, float)) else None
    start = config.get("START_FREQ")
    bandwidth = config.get("B")
    end = start + bandwidth if all(isinstance(v, (int, float)) for v in (start, bandwidth)) else None

    return {
        "chirp_config_hash": hashlib.sha256(canonical).hexdigest(),
        "a0_compatible_chirp_config_hash": hashlib.sha256(a0_canonical).hexdigest()[:16],
        "canonical_json": canonical.decode("utf-8"),
        "original_fields": config,
        "field_interpretations": field_records,
        "interpreted": {
            "start_frequency_hz": start,
            "sampled_bandwidth_hz": bandwidth,
            "sampled_end_frequency_hz": end,
            "sampled_center_frequency_hz": (start + end) / 2
            if isinstance(start, (int, float)) and isinstance(end, (int, float))
            else None,
            "adc_samples": config.get("ADC_SAMPLES"),
            "range_fft_size": None,
            "configured_loop_count": config.get("LOOPS"),
            "chirps_per_frame": None,
            "rx_antenna_count": rx,
            "tx_antenna_count": tx,
            "virtual_antenna_count": virtual_count,
            "frame_period_seconds": frame_period,
            "configured_frame_rate_hz": 1.0 / frame_period if frame_period else None,
            "range_bin_spacing_m": config.get("R_BIN"),
            "maximum_range_m": config.get("R_MAX"),
            "channel_ordering": None,
        },
        "unresolved": [
            "Chirp ordering and TX/RX-to-virtual-channel ordering are absent from the config.",
            "LOOPS is preserved but chirps_per_frame is not asserted without producer details.",
            "The config has ADC_SAMPLES but no explicit range FFT size field.",
        ],
    }


def classify_alignment(frame_count: int | None, timestamp_count: int | None) -> tuple[str, int | None]:
    if frame_count is None:
        return "DECODE_FAILURE", None
    if timestamp_count is None:
        return "TIMESTAMP_PARSE_FAILURE", None
    difference = frame_count - timestamp_count
    if difference == 0:
        return "EXACT_ALIGNMENT", difference
    if abs(difference) == 1:
        return "OFF_BY_ONE", difference
    return "FRAME_COUNT_MISMATCH", difference


class SafeRFFTReader:
    """Read one recording from ZIP into a structural tensor contract."""

    def __init__(
        self,
        *,
        max_compressed_bytes: int = DEFAULT_MAX_COMPRESSED_BYTES,
        max_decompressed_bytes: int = DEFAULT_MAX_DECOMPRESSED_BYTES,
        max_text_bytes: int = DEFAULT_MAX_TEXT_BYTES,
    ) -> None:
        self.max_compressed_bytes = max_compressed_bytes
        self.max_decompressed_bytes = max_decompressed_bytes
        self.max_text_bytes = max_text_bytes

    def _read_text_member(self, archive: zipfile.ZipFile, member: str) -> bytes:
        info = archive.getinfo(member)
        if info.file_size > self.max_text_bytes:
            raise RFFTReaderError(
                f"text member {member} exceeds {self.max_text_bytes} bytes"
            )
        with archive.open(info, "r") as stream:
            raw = stream.read(self.max_text_bytes + 1)
        if len(raw) > self.max_text_bytes:
            raise RFFTReaderError(
                f"text member {member} exceeds {self.max_text_bytes} bytes"
            )
        return raw

    def read_recording(
        self,
        *,
        archive_path: str,
        radar_member: str,
        timestamp_member: str,
        chirp_config_member: str,
    ) -> dict[str, Any]:
        warnings = [
            "OBJECT_EXECUTION_CAPABLE_PICKLE_CONTAINER_DECODED_BY_RESTRICTED_NON_EXECUTING_VM"
        ]
        errors: list[str] = []
        tensor: np.ndarray | None = None
        range_bins: np.ndarray | None = None
        structural: dict[str, Any] = {}
        timestamp_metadata: dict[str, Any] = {}
        chirp_metadata: dict[str, Any] = {}

        with zipfile.ZipFile(archive_path, "r") as archive:
            radar_info = archive.getinfo(radar_member)
            if radar_info.file_size > self.max_compressed_bytes:
                raise DecompressionError(
                    f"ZIP member payload exceeds {self.max_compressed_bytes} bytes"
                )
            with archive.open(radar_info, "r") as radar_stream:
                payload, decompression = bounded_zlib_decompress(
                    radar_stream,
                    max_compressed_bytes=self.max_compressed_bytes,
                    max_decompressed_bytes=self.max_decompressed_bytes,
                )
            tensor, range_bins, serialization = decode_restricted_numpy_pickle(payload)
            timestamps_raw = self._read_text_member(archive, timestamp_member)
            config_raw = self._read_text_member(archive, chirp_config_member)
            timestamp_metadata = parse_radar_timestamps(timestamps_raw)
            chirp_metadata = parse_chirp_config(config_raw)

        frame_count = int(tensor.shape[0])
        alignment, difference = classify_alignment(
            frame_count, timestamp_metadata["timestamp_count"]
        )
        byteorder = tensor.dtype.byteorder
        endianness = (
            "little"
            if byteorder == "<" or (byteorder == "=" and sys.byteorder == "little")
            else "big"
        )
        structural = {
            **decompression,
            **serialization,
            "zip_compression_method": radar_info.compress_type,
            "zip_compressed_size_bytes": radar_info.compress_size,
            "payload_decode_status": "SUCCESS_WITH_WARNING",
            "dtype": tensor.dtype.name,
            "dtype_string": tensor.dtype.str,
            "endianness": endianness,
            "is_complex": bool(np.issubdtype(tensor.dtype, np.complexfloating)),
            "complex_representation": "NATIVE_NUMPY_COMPLEX_INTERLEAVED_REAL_IMAG_FLOAT64",
            "shape": list(tensor.shape),
            "range_bins_shape": list(range_bins.shape),
            "range_bins_dtype": range_bins.dtype.name,
            "range_bins_first_m": float(range_bins[0]),
            "range_bins_last_m": float(range_bins[-1]),
            "range_bins_spacing_m_median": float(np.median(np.diff(range_bins))),
            "frame_axis": 0,
            "antenna_axis": 1,
            "range_bin_axis": 2,
            "axis_semantics_evidence": {
                "frame_axis": [
                    "OFFICIAL_DATASET_SOURCE_CODE",
                    "INFERRED_FROM_FRAME_TIMESTAMP_DIMENSION_CONSISTENCY",
                ],
                "antenna_axis": [
                    "OFFICIAL_DATASET_DOCUMENTATION",
                    "DIRECT_CHIRP_CONFIG",
                ],
                "range_bin_axis": [
                    "OFFICIAL_DATASET_DOCUMENTATION",
                    "DIRECT_PAYLOAD_STRUCTURE",
                    "DIRECT_CHIRP_CONFIG",
                ],
            },
            "virtual_antenna_count": int(tensor.shape[1]),
            "virtual_antenna_ordering": None,
            "range_bin_count": int(tensor.shape[2]),
            "frame_count": frame_count,
            "frame_timestamp_difference": difference,
            "alignment_status": alignment,
        }
        interpreted = chirp_metadata["interpreted"]
        if interpreted["virtual_antenna_count"] != structural["virtual_antenna_count"]:
            errors.append("VIRTUAL_ANTENNA_COUNT_CONFIG_MISMATCH")
        if interpreted["adc_samples"] != structural["range_bin_count"]:
            warnings.append("ADC_SAMPLE_COUNT_RANGE_BIN_COUNT_DIFFERENCE")
        configured_spacing = interpreted["range_bin_spacing_m"]
        measured_spacing = structural["range_bins_spacing_m_median"]
        if isinstance(configured_spacing, (int, float)) and not math.isclose(
            configured_spacing, measured_spacing, rel_tol=1e-9, abs_tol=1e-12
        ):
            warnings.append("CONFIGURED_R_BIN_DIFFERS_FROM_STORED_RBINS_SPACING")
        if alignment != "EXACT_ALIGNMENT":
            warnings.append(alignment)
        if errors:
            structural["payload_decode_status"] = "FAILURE"

        return {
            "tensor": tensor,
            "range_bins": range_bins,
            "structural_metadata": structural,
            "timestamp_metadata": timestamp_metadata,
            "chirp_metadata": chirp_metadata,
            "warnings": warnings,
            "errors": errors,
        }


__all__ = [
    "ChirpConfigError",
    "DecompressionError",
    "PayloadFormatError",
    "RFFTReaderError",
    "SafeRFFTReader",
    "TimestampParseError",
    "UnsafeSerializationError",
    "bounded_zlib_decompress",
    "bounded_zlib_decompress_bytes",
    "classify_alignment",
    "decode_restricted_numpy_pickle",
    "identify_payload_format",
    "parse_chirp_config",
    "parse_radar_timestamps",
]
