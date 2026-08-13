# Standalone Repository Publication Audit

## Scope

This initial publication uses the current local `safenest_integration` directory as its source of truth. It does not fetch or merge new SafeNest source changes, retrain models, or promote candidate artifacts. The original `safenest_system_integration/` directory and ZIP remain outside this repository and were not modified.

## Candidate identity

- Target repository: `https://github.com/yuname121/integration.git`
- Candidate root: the contents of `safenest_integration/` copied into this repository root
- Source snapshot provenance: `LATEST_SOURCE_PROVENANCE.json`
- Source files in the candidate: 1,158
- Candidate payload size before Git metadata: approximately 112 MB
- Largest individual file: approximately 25.75 MB
- Files over GitHub's 100 MB single-file limit: none observed

## Standalone boundary changes

The original integration folder was designed to run as a package named `safenest_integration` from its parent repository. For this repository, only the staging copy was minimally adapted so the repository root is directly executable:

- Python imports now resolve to root packages such as `ai`, `backend`, `gateway`, `database`, `hil`, and `state`.
- `deployment/run_pi.sh` runs `hil.preflight` and `backend/run_backend.py` from the repository root.
- Documentation commands use `cd ~/integration` and root-relative paths.
- The frozen `sources/ondevice_ai/` snapshot was not edited.

## Safety checks

- No `secrets.h`, real `.env`, SSH key, token, password, or API credential detected.
- No SQLite database, Python virtual environment, cache, or `__pycache__` included.
- `.gitignore` excludes credentials, databases, virtual environments, logs, caches, OS metadata, and ZIP files.
- Required model and firmware files are present.
- Model SHA-256 checks match the recorded manifest values.

## Validation

- Root-level import and regression suite: 110 tests passed.
- Bundle verifier: passed; required files present, model hashes match, no secrets/databases/caches.
- Hardware validation: not performed in this publication environment.
- Raspberry Pi, ESP32, MR60, SCD40, and Thermal-44 claims remain subject to physical HIL validation.

## Remote preflight

The target remote returned no branch heads during pre-publication inspection, so it appears empty. No remote history was overwritten. The publication agent must still verify the remote immediately before the first push and stop if meaningful history appears.
