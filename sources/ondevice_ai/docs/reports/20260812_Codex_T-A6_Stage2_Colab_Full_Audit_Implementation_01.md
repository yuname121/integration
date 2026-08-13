# SafeNest Thermal T-A6 Stage 2 — Colab Full-Audit Implementation

Phase: `T-A6`

This change completes the guarded Stage 2 execution package that runs only in
owner-started Google Colab with the SDT payload materialized in Google Drive.
It does not train a model, create a new split, or authorize T-B.

## Execution contract

The runner requires the exact source payload names `train.zip.001` through
`train.zip.004`, `validation.zip`, and `test.zip` when execution is requested.
Startup inspection remains bounded and non-hydrating.  The train parts are
accepted only as a `RAW_BYTE_SPLIT_CANDIDATE`; independent ZIP pieces and
unknown formats fail closed.  The logical train archive is reconstructed and
validated before any member conversion.

Synthetic TRAIN and VALIDATION source contracts are independently checked for
the documented `640x480` 16-bit grayscale `image_t`/`image_d` pairs, labels,
Kelvin-centiunit Thermal encoding, and millimetre depth representation.  The
real `test.zip` is exposed only through the locked reader path and remains
`REAL_EVAL_DEVELOPMENT`; it is never promoted to `LOCKED_TEST`.

## Stage 2 evidence

The persistent `T-A6_execution_result/` bundle contains deterministic compact
JSON evidence for:

- source identity and synthetic physical-contract verification;
- TRAIN, VALIDATION, and REAL_EVAL_DEVELOPMENT artifact/provenance registries;
- explicit source-frame status reconciliation and quality accounting;
- exact source-byte, decoded-frame, and canonical-frame duplicate audits;
- frozen-profile within-role and cross-role near-duplicate screening;
- measurable leakage plus explicit NOT_VERIFIABLE subject/session/sequence/event
  limitations;
- full second-conversion checksum replay;
- portable paths and checksum coverage.

Bulk tensors and provenance remain in the owner-selected persistent output
location and are not tracked in Git.  The standalone
`scripts/validate_thermal_t_a6_stage2.py` validator reads only the compact
bundle and returns `PASS_WITH_LIMITATIONS` when the evidence is complete.  It
always reports `t_b_authorized: false`.

## Colab invocation

After the branch containing this implementation is merged, checkout the exact
commit shown by the PR and run a dry-run with `--include-real-test`.  Only when
that guard passes should the owner add `--execute`:

```bash
python scripts/run_thermal_t_a6_colab.py \
  --mode COLAB_STAGE2 \
  --drive-raw-root /content/drive/MyDrive/SafeNest \
  --work-root /content/thermal_t_a6_work \
  --drive-output-root /content/drive/MyDrive/SafeNest/T-A6 \
  --repo-root /content/safenest-t-a6 \
  --include-real-test \
  --execute
```

The output `FULL_AUDIT_COMPLETE_WITH_LIMITATIONS` is not a model result and is
not T-B authorization.  The compact bundle must be returned for the final
Thermal T-A6 validator review.

## Full validator closure

`scripts/validate_thermal_t_a6.py --mode FULL_DATASET` now accepts the compact
`T-A6_execution_result/` directory as its evidence root.  It live-calls the
Stage-2 compact validator and independently runs the T-A0 through T-A5
predecessor chain; it does not trust a previously persisted
`validation_result.json` as proof of current validity.  The full gate remains
`T_A6_FULL_COMPLETE_WITH_LIMITATIONS` and `t_b_authorized` remains `false`.
