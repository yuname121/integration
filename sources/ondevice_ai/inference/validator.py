#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inference/validator.py
SafeNest active-workspace TFLite Model & Manifest Ground Truth Validator

Extracts actual tensor metadata from .tflite model files using TFLite interpreter
(Ground Truth) and cross-validates with model_manifest.json, config/models.yaml,
and runtime interpreter configurations.
"""

from __future__ import annotations
from pathlib import Path
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Rationale on Class Label Verification:
# Standard TFLite model flatbuffers do not natively store custom string class labels in their
# schema unless embedded as custom metadata buffers. Therefore, model_manifest.json serves as the
# declared source of truth for ordered class labels. The validator strictly cross-verifies that:
# 1. TFLite output tensor dimension [1, C] exactly equals the number of classes C declared in manifest.
# 2. Manifest class label order (0 -> C-1) matches config (config/models.yaml) and runtime label mappings.
# 3. Any discrepancy in class count or class ordering causes a validation failure.


class DependencyError(RuntimeError):
    """Raised when TFLite runtime or TensorFlow dependency is missing."""
    pass


class ConfigValidationError(ValueError):
    """Raised when configuration or model validation fails."""
    pass


def find_repo_root(start_path: Optional[str | Path] = None) -> Path:
    """
    Finds the repository root without assuming a particular project version name.
    Normalizes path to absolute Path object without resolving symlinks dangerously.
    """
    current = Path(start_path).resolve() if start_path else Path.cwd().resolve()
    for p in [current] + list(current.parents):
        if (p / "AGENTS.md").is_file() and (p / "models" / "model_manifest.json").is_file():
            return p
    # Standalone package copies may not contain the workspace instruction file.
    file_dir = Path(__file__).resolve().parent
    if file_dir.name == "inference" and (file_dir.parent / "models" / "model_manifest.json").is_file():
        return file_dir.parent
    return current


def load_tflite_interpreter_class():
    """Dynamically imports TFLite Interpreter, or raises DependencyError with install instructions."""
    try:
        import ai_edge_litert.interpreter as tflite
        return tflite.Interpreter
    except ImportError:
        pass
    try:
        import tflite_runtime.interpreter as tflite
        return tflite.Interpreter
    except ImportError:
        pass
    try:
        import tensorflow.lite as tflite
        return tflite.Interpreter
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.lite.Interpreter
    except ImportError:
        pass

    raise DependencyError(
        "TFLite runtime or TensorFlow dependency is missing. "
        "Please install 'tflite-runtime' (pip install tflite-runtime) "
        "or 'tensorflow' (pip install tensorflow) to run model verification."
    )


def normalize_dtype_name(dtype_val: Any) -> str:
    """Normalizes numpy / python / string dtype representation into standard string (e.g. 'int8', 'float32')."""
    if isinstance(dtype_val, str):
        s = dtype_val.strip().lower()
        if s.startswith("numpy."):
            s = s[6:]
        if s.startswith("<class 'numpy.") and s.endswith("'>"):
            s = s[14:-2]
        return s
    return np.dtype(dtype_val).name


def normalize_repo_relative_path(path_str: str, project_root: Path, context: str = "") -> str:
    """
    Verifies and normalizes a path relative to the active project root.
    Fails if path is absolute, escapes repository (with ..), or uses home (~).
    """
    if not isinstance(path_str, str) or not path_str.strip():
        raise ConfigValidationError(f"[{context}] Path string is empty or invalid: {path_str!r}")

    cleaned = path_str.strip()
    if cleaned.startswith("/") or cleaned.startswith("\\") or (len(cleaned) > 1 and cleaned[1] == ":"):
        raise ConfigValidationError(
            f"[{context}] Absolute path is forbidden in config/manifest: {cleaned!r}"
        )
    if cleaned.startswith("~"):
        raise ConfigValidationError(
            f"[{context}] Home directory expansion '~' is forbidden in config/manifest: {cleaned!r}"
        )

    root = project_root.resolve()
    path_obj = Path(cleaned)
    if path_obj.parts and path_obj.parts[0] == root.name:
        path_obj = Path(*path_obj.parts[1:])
    target = (root / path_obj).resolve()

    try:
        rel = target.relative_to(root)
        rel_str = rel.as_posix()
    except ValueError:
        raise ConfigValidationError(
            f"[{context}] Path escapes repository root: {cleaned!r} -> {target}"
        )

    return rel_str


def compare_floats(val1: float, val2: float, atol: float = 1e-6, context: str = "") -> Tuple[bool, str]:
    """Compares two float numbers with explicit absolute tolerance atol=1e-6."""
    diff = abs(val1 - val2)
    is_equal = diff <= atol
    msg = f"expected {val2}, actual {val1} (diff={diff:.8f}, atol={atol})"
    return is_equal, msg


class GroundTruthValidator:
    def __init__(
        self,
        repo_root: Optional[Path] = None,
        project_root: Optional[Path] = None,
    ):
        module_project_root = Path(__file__).resolve().parent.parent
        explicit_root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()

        if project_root is not None:
            active_project_root = Path(project_root).resolve()
        elif (explicit_root / "models" / "model_manifest.json").is_file():
            active_project_root = explicit_root
        elif (explicit_root / module_project_root.name / "models" / "model_manifest.json").is_file():
            active_project_root = explicit_root / module_project_root.name
        else:
            active_project_root = module_project_root

        self.project_root = active_project_root.resolve()
        self.repo_root = (
            explicit_root
            if explicit_root != self.project_root
            else self.project_root.parent
        )
        self.errors: List[str] = []

    def _display_path(self, relative_path: str) -> str:
        return f"{self.project_root.name}/{relative_path}"

    def _log_error(self, sensor_or_model: str, test_item: str, expected: Any, actual: Any):
        msg = f"❌ [{sensor_or_model}] {test_item} - Expected: {expected!r}, Actual: {actual!r}"
        self.errors.append(msg)

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Loads YAML using PyYAML if available, or a fallback parser for simple YAML structures."""
        text = path.read_text(encoding="utf-8")
        try:
            import yaml
            return yaml.safe_load(text) or {}
        except ImportError:
            result: Dict[str, Any] = {}
            current_model = None
            current_section = None

            for line in text.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                indent = len(line) - len(line.lstrip())
                if ":" not in stripped:
                    continue

                k, v = stripped.split(":", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")

                if indent == 2:  # thermal, mmwave, co2
                    current_model = k
                    current_section = None
                    if "models" not in result:
                        result["models"] = {}
                    result["models"][current_model] = {}
                elif indent == 4 and current_model:  # input, output, class_map
                    if v == "":
                        current_section = k
                        result["models"][current_model][current_section] = {}
                    else:
                        result["models"][current_model][k] = self._parse_yaml_val(v)
                elif indent == 6 and current_model and current_section:
                    parsed_v = self._parse_yaml_val(v)
                    try:
                        k_key = int(k)
                    except ValueError:
                        k_key = k
                    result["models"][current_model][current_section][k_key] = parsed_v

            return result

    @staticmethod
    def _parse_yaml_val(v: str) -> Any:
        if v.startswith("[") and v.endswith("]"):
            elements = [x.strip().strip('"').strip("'") for x in v[1:-1].split(",") if x.strip()]
            parsed_list = []
            for elem in elements:
                try:
                    parsed_list.append(int(elem))
                except ValueError:
                    try:
                        parsed_list.append(float(elem))
                    except ValueError:
                        parsed_list.append(elem)
            return parsed_list
        if v.lower() == "true":
            return True
        if v.lower() == "false":
            return False
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v

    def validate_all(self, generate_inventory: bool = True) -> Tuple[bool, Dict[str, Any], List[str]]:
        self.errors.clear()
        inventory: Dict[str, Any] = {
            "schema_version": "1.0",
            "project": "SafeNest",
            "models": {}
        }

        # 1. Load Interpreter Class
        try:
            InterpreterClass = load_tflite_interpreter_class()
        except DependencyError as e:
            self.errors.append(f"❌ [DEPENDENCY_ERROR] {e}")
            return False, inventory, self.errors

        # 2. Check and load manifest
        manifest_project_rel = "models/model_manifest.json"
        manifest_rel = self._display_path(manifest_project_rel)
        manifest_abs = self.project_root / manifest_project_rel
        if not manifest_abs.is_file():
            self.errors.append(f"❌ [MANIFEST] Manifest file missing at {manifest_rel}")
            return False, inventory, self.errors

        try:
            manifest_text = manifest_abs.read_text(encoding="utf-8")
            manifest = json.loads(manifest_text)
        except Exception as e:
            self.errors.append(f"❌ [MANIFEST] Failed to parse manifest JSON: {e}")
            return False, inventory, self.errors

        # Check for absolute paths inside manifest raw text
        def check_json_abs_paths(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    check_json_abs_paths(v, f"{path}.{k}")
            elif isinstance(obj, str):
                if obj.startswith("/") or obj.startswith("~") or (len(obj) > 1 and obj[1] == ":"):
                    self.errors.append(f"❌ [MANIFEST] Absolute path detected in manifest key {path}: {obj!r}")
        check_json_abs_paths(manifest)

        # 3. Check and load config/models.yaml (if present)
        config_project_rel = "config/models.yaml"
        config_rel = self._display_path(config_project_rel)
        config_abs = self.project_root / config_project_rel
        config_data = {}
        if config_abs.is_file():
            config_data = self._load_yaml(config_abs)

        # Check for absolute paths inside config_data
        def check_config_abs_paths(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    check_config_abs_paths(v, f"{path}.{k}")
            elif isinstance(obj, str):
                if obj.startswith("/") or obj.startswith("~") or (len(obj) > 1 and obj[1] == ":"):
                    self.errors.append(f"❌ [CONFIG] Absolute path detected in config key {path}: {obj!r}")
        check_config_abs_paths(config_data)

        models_manifest = manifest.get("models", {})
        target_model_keys = ["thermal", "mmwave", "co2"]

        for model_key in target_model_keys:
            m_meta = models_manifest.get(model_key)
            if not m_meta:
                self._log_error(model_key, "manifest entry", "model entry in manifest", "None")
                continue

            model_id = m_meta.get("model_id", model_key)
            sensor_id = model_key

            # Validate paths in manifest
            try:
                manifest_model_path_rel = normalize_repo_relative_path(
                    m_meta.get("path", ""), self.project_root, f"{model_id}.manifest.path"
                )
                full_model_rel = self._display_path(manifest_model_path_rel)
                full_model_abs = self.project_root / manifest_model_path_rel
            except ConfigValidationError as e:
                self.errors.append(f"❌ [{model_id}] {e}")
                full_model_rel = m_meta.get("path", "")
                full_model_abs = self.project_root / full_model_rel

            # Validate config model path if available
            config_models = config_data.get("models", {})
            cfg_m = config_models.get(model_key, {})
            cfg_model_rel = None
            if cfg_m and "path" in cfg_m:
                try:
                    cfg_path_norm = normalize_repo_relative_path(cfg_m["path"], self.project_root, f"{model_id}.config.path")
                    cfg_model_rel = self._display_path(cfg_path_norm)
                except ConfigValidationError as e:
                    self.errors.append(f"❌ [{model_id}] {e}")
                    cfg_model_rel = cfg_m["path"]

                if cfg_model_rel != full_model_rel:
                    self._log_error(model_id, "path consistency (manifest vs config)", full_model_rel, cfg_model_rel)

            # File existence
            if not full_model_abs.is_file():
                self._log_error(model_id, "model file existence", f"file at {full_model_rel}", "FILE_NOT_FOUND")
                model_bytes = b""
                actual_sha256 = "FILE_MISSING"
                file_size = 0
            else:
                model_bytes = full_model_abs.read_bytes()
                actual_sha256 = hashlib.sha256(model_bytes).hexdigest()
                file_size = len(model_bytes)

            expected_sha256 = m_meta.get("sha256")
            if expected_sha256 and actual_sha256 != expected_sha256:
                self._log_error(model_id, "SHA256 checksum", expected_sha256, actual_sha256)

            # Metadata / Scaler paths check
            metadata_path_rel = None
            if "metadata_path" in m_meta:
                try:
                    meta_norm = normalize_repo_relative_path(m_meta["metadata_path"], self.project_root, f"{model_id}.metadata_path")
                    metadata_path_rel = self._display_path(meta_norm)
                    if not (self.project_root / meta_norm).is_file():
                        self._log_error(model_id, "scaler/metadata file existence", metadata_path_rel, "FILE_NOT_FOUND")
                except ConfigValidationError as e:
                    self.errors.append(f"❌ [{model_id}] {e}")
                    metadata_path_rel = m_meta["metadata_path"]

            # Load TFLite Interpreter Ground Truth
            model_inventory_entry: Dict[str, Any] = {
                "model_id": model_id,
                "sensor_id": sensor_id,
                "repository_relative_model_path": full_model_rel,
                "sha256": actual_sha256,
                "size_bytes": file_size,
                "manifest_path": manifest_rel,
                "config_path": config_rel if config_abs.is_file() else None,
                "runtime_resolved_path": full_model_rel,
                "scaler_or_metadata_path": metadata_path_rel,
                "validation_status": "PENDING",
                "errors": []
            }

            if not full_model_abs.is_file():
                model_inventory_entry["validation_status"] = "FAILED"
                model_inventory_entry["errors"] = [f"Model file missing: {full_model_rel}"]
                inventory["models"][sensor_id] = model_inventory_entry
                continue

            try:
                interpreter = InterpreterClass(model_path=str(full_model_abs))
                interpreter.allocate_tensors()

                inputs_detail = interpreter.get_input_details()
                outputs_detail = interpreter.get_output_details()

                if not inputs_detail or not outputs_detail:
                    raise ValueError("Interpreter returned empty input or output details")

                inp = inputs_detail[0]
                out = outputs_detail[0]

                # Extract Ground Truth
                inp_name = inp["name"]
                inp_shape = inp["shape"].tolist()
                inp_signature = inp.get("shape_signature", inp["shape"]).tolist()
                inp_dtype = normalize_dtype_name(inp["dtype"])
                inp_scale, inp_zero_point = inp["quantization"]
                inp_scale = float(inp_scale)
                inp_zero_point = int(inp_zero_point)

                out_name = out["name"]
                out_shape = out["shape"].tolist()
                out_signature = out.get("shape_signature", out["shape"]).tolist()
                out_dtype = normalize_dtype_name(out["dtype"])
                out_scale, out_zero_point = out["quantization"]
                out_scale = float(out_scale)
                out_zero_point = int(out_zero_point)

                # Store ground truth tensor details in inventory
                model_inventory_entry["input_tensor"] = {
                    "name": inp_name,
                    "shape": inp_shape,
                    "shape_signature": inp_signature,
                    "dtype": inp_dtype,
                    "scale": inp_scale,
                    "zero_point": inp_zero_point,
                }
                model_inventory_entry["output_tensor"] = {
                    "name": out_name,
                    "shape": out_shape,
                    "shape_signature": out_signature,
                    "dtype": out_dtype,
                    "scale": out_scale,
                    "zero_point": out_zero_point,
                }

                # Cross-validate Manifest Input Contract
                m_inp = m_meta.get("input", {})
                if "name" in m_inp and m_inp["name"] != inp_name:
                    self._log_error(model_id, "input tensor name", m_inp["name"], inp_name)
                if m_inp.get("shape") != inp_shape:
                    self._log_error(model_id, "input shape", m_inp.get("shape"), inp_shape)

                m_inp_dtype = normalize_dtype_name(m_inp.get("dtype", ""))
                if m_inp_dtype != inp_dtype:
                    self._log_error(model_id, "input dtype", m_inp_dtype, inp_dtype)

                if "scale" in m_inp:
                    m_scale = float(m_inp["scale"])
                    ok, msg = compare_floats(inp_scale, m_scale, atol=1e-6)
                    if not ok:
                        self._log_error(model_id, "input quantization scale", m_scale, f"{inp_scale} ({msg})")
                if "zero_point" in m_inp:
                    if int(m_inp["zero_point"]) != inp_zero_point:
                        self._log_error(model_id, "input zero point", m_inp["zero_point"], inp_zero_point)

                # Cross-validate Manifest Output Contract
                m_out = m_meta.get("output", {})
                if "name" in m_out and m_out["name"] != out_name:
                    self._log_error(model_id, "output tensor name", m_out["name"], out_name)
                if m_out.get("shape") != out_shape:
                    self._log_error(model_id, "output shape", m_out.get("shape"), out_shape)

                m_out_dtype = normalize_dtype_name(m_out.get("dtype", ""))
                if m_out_dtype != out_dtype:
                    self._log_error(model_id, "output dtype", m_out_dtype, out_dtype)

                if "scale" in m_out:
                    m_scale = float(m_out["scale"])
                    ok, msg = compare_floats(out_scale, m_scale, atol=1e-6)
                    if not ok:
                        self._log_error(model_id, "output quantization scale", m_scale, f"{out_scale} ({msg})")
                if "zero_point" in m_out:
                    if int(m_out["zero_point"]) != out_zero_point:
                        self._log_error(model_id, "output zero point", m_out["zero_point"], out_zero_point)

                # Cross-validate Config Input/Output Contract (if present)
                if cfg_m:
                    cfg_inp = cfg_m.get("input", {})
                    if cfg_inp.get("shape") and cfg_inp["shape"] != inp_shape:
                        self._log_error(model_id, "config input shape", inp_shape, cfg_inp["shape"])
                    if cfg_inp.get("dtype") and normalize_dtype_name(cfg_inp["dtype"]) != inp_dtype:
                        self._log_error(model_id, "config input dtype", inp_dtype, cfg_inp["dtype"])
                    if "scale" in cfg_inp:
                        ok, msg = compare_floats(inp_scale, float(cfg_inp["scale"]), atol=1e-6)
                        if not ok:
                            self._log_error(model_id, "config input scale", inp_scale, cfg_inp["scale"])
                    if "zero_point" in cfg_inp and int(cfg_inp["zero_point"]) != inp_zero_point:
                        self._log_error(model_id, "config input zero point", inp_zero_point, cfg_inp["zero_point"])

                    cfg_out = cfg_m.get("output", {})
                    if cfg_out.get("shape") and cfg_out["shape"] != out_shape:
                        self._log_error(model_id, "config output shape", out_shape, cfg_out["shape"])

                # Class Verification
                if len(out_shape) >= 2:
                    inferred_class_count = out_shape[-1]
                else:
                    inferred_class_count = out_shape[0]

                class_map = m_meta.get("class_map", {})
                manifest_class_count = len(class_map)

                if inferred_class_count != manifest_class_count:
                    self._log_error(
                        model_id,
                        "class count mismatch",
                        f"output tensor dim={inferred_class_count}",
                        f"manifest count={manifest_class_count}"
                    )

                # Check class order and key mapping (0..C-1)
                sorted_class_keys = sorted([int(k) for k in class_map.keys()])
                expected_keys = list(range(manifest_class_count))
                if sorted_class_keys != expected_keys:
                    self._log_error(model_id, "class index sequence", expected_keys, sorted_class_keys)

                ordered_class_labels = [class_map[str(i)] for i in expected_keys if str(i) in class_map]

                # Check config class_map consistency
                if cfg_m and "class_map" in cfg_m:
                    cfg_class_map = cfg_m["class_map"]
                    cfg_ordered_labels = [str(cfg_class_map[i]) for i in sorted([int(k) for k in cfg_class_map.keys()])]
                    if cfg_ordered_labels != ordered_class_labels:
                        self._log_error(
                            model_id,
                            "class order / label mismatch (manifest vs config)",
                            ordered_class_labels,
                            cfg_ordered_labels
                        )

                model_inventory_entry["class_count"] = manifest_class_count
                model_inventory_entry["ordered_class_labels"] = ordered_class_labels

                # Check runtime interpreter class_map
                try:
                    sys_path_added = False
                    sn_dir = str(self.project_root)
                    if sn_dir not in sys.path:
                        sys.path.insert(0, sn_dir)
                        sys_path_added = True

                    if sensor_id == "thermal":
                        from inference.thermal_interpreter import ThermalInterpreter
                        rt = ThermalInterpreter(project_root=sn_dir, manifest_path="models/model_manifest.json")
                        rt_labels = [rt.class_map[i] for i in sorted(rt.class_map.keys())]
                        if rt_labels != ordered_class_labels:
                            self._log_error(model_id, "runtime label order mismatch", ordered_class_labels, rt_labels)
                    elif sensor_id == "co2":
                        from inference.co2_interpreter import CO2Interpreter
                        rt = CO2Interpreter(project_root=sn_dir, manifest_path="models/model_manifest.json")
                        rt_labels = [rt.class_map[i] for i in sorted(rt.class_map.keys())]
                        if rt_labels != ordered_class_labels:
                            self._log_error(model_id, "runtime label order mismatch", ordered_class_labels, rt_labels)
                    elif sensor_id == "mmwave":
                        from inference.mmwave_interpreter import MMWaveInterpreter
                        rt = MMWaveInterpreter(project_root=sn_dir, manifest_path="models/model_manifest.json")
                        rt_labels = [rt.class_map[i] for i in sorted(rt.class_map.keys())]
                        if rt_labels != ordered_class_labels:
                            self._log_error(model_id, "runtime label order mismatch", ordered_class_labels, rt_labels)
                except Exception as rt_err:
                    self.errors.append(f"❌ [{model_id}] Runtime interpreter initialization failed: {rt_err}")

                model_inventory_entry["validation_status"] = "PASSED" if not self.errors else "FAILED"

            except Exception as e:
                self.errors.append(f"❌ [{model_id}] TFLite interpreter validation exception: {e}")
                model_inventory_entry["validation_status"] = "FAILED"
                model_inventory_entry["errors"].append(str(e))

            inventory["models"][sensor_id] = model_inventory_entry

        is_valid = len(self.errors) == 0

        if generate_inventory:
            report_rel = "docs/reports/model_inventory.json"
            report_abs = self.project_root / report_rel
            report_abs.parent.mkdir(parents=True, exist_ok=True)
            # Write deterministic JSON with sorted keys
            report_abs.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        return is_valid, inventory, self.errors


def validate_active_config(
    repo_root: Optional[Path] = None,
    project_root: Optional[Path] = None,
    generate_inventory: bool = True,
) -> bool:
    """Convenience function to run validation and print error summary if failed."""
    validator = GroundTruthValidator(repo_root=repo_root, project_root=project_root)
    is_valid, _, errors = validator.validate_all(generate_inventory=generate_inventory)
    if not is_valid:
        print("❌ SafeNest active config & ground truth validation failed:", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
    return is_valid


if __name__ == "__main__":
    success = validate_active_config()
    sys.exit(0 if success else 1)
