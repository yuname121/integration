# SafeNest mmWave M-B7 — Deterministic Input-Perturbation Robustness

- Phase: `M-B7`
- Scope: `OFFLINE_REAL_DATA_PERTURBATION_ROBUSTNESS`
- Frozen architecture: `M-B3_CONV1D_GAP_BASELINE`
- Frozen seeds: `[42, 43, 44]`
- Frozen calibration: `M-B5_CAL_CLASS_BALANCED_120`
- Evaluation population: 79 pure-class VALIDATION windows / 17 subjects
- LOCKED_TEST performance access: `0`
- Model trainings: `0`
- Model conversions: `0`

## Clean M-B6 identity

Fresh strict-INT8 clean inference reproduced the M-B6 top-1 vectors, per-class metrics,
Macro F1, accuracy, input saturation, and output endpoint ratio for all three seeds.

| Seed | Macro F1 | Accuracy | Prediction distribution N/R/A | Input saturation | Output endpoint |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.666231 | 0.721519 | 12 / 14 / 53 | 0.000000000 | 0.000000000 |
| 43 | 0.441240 | 0.455696 | 23 / 35 / 21 | 0.000000000 | 0.000000000 |
| 44 | 0.329107 | 0.518987 | 7 / 1 / 71 | 0.000000000 | 0.000000000 |

## Preregistered perturbations and cross-seed worst cases

| Profile | Worst F1 degradation | Worst recall degradation | Minimum Top-1 | Max saturation | Worst confidence change | Collapsed seeds |
|---|---:|---:|---:|---:|---:|---:|
| `M-B7_GAUSSIAN_SNR20` | 0.106140 (seed 42) | 0.100000 | 0.886076 | 0.000000000 | -0.007120 | 0 |
| `M-B7_GAUSSIAN_SNR10` | 0.095688 (seed 42) | 0.136363 | 0.797468 | 0.001729958 | -0.030311 | 0 |
| `M-B7_GAUSSIAN_POST_B1_SNR20` | 0.089725 (seed 42) | 0.135135 | 0.848101 | 0.000000000 | -0.028679 | 0 |
| `M-B7_GAUSSIAN_POST_B1_SNR10` | 0.156787 (seed 42) | 0.567568 | 0.329114 | 0.000042194 | -0.035403 | 0 |
| `M-B7_AMP_X0_50` | 0.421275 (seed 42) | 0.500000 | 0.354430 | 0.000000000 | -0.003362 | 2 |
| `M-B7_AMP_X0_75` | 0.177057 (seed 42) | 0.350000 | 0.632911 | 0.000000000 | -0.003659 | 1 |
| `M-B7_AMP_X1_25` | 0.103001 (seed 42) | 0.227272 | 0.721519 | 0.000168776 | -0.023685 | 0 |
| `M-B7_AMP_X1_50` | 0.150688 (seed 42) | 0.272727 | 0.658228 | 0.001434599 | -0.033970 | 0 |
| `M-B7_DRIFT_MILD` | 0.022130 (seed 42) | 0.050000 | 0.974684 | 0.000000000 | -0.000890 | 0 |
| `M-B7_DRIFT_SEVERE` | 0.024272 (seed 42) | 0.045454 | 0.949367 | 0.000000000 | -0.001780 | 0 |
| `M-B7_DROPOUT_SHORT` | 0.000000 (seed 42) | 0.000000 | 0.987342 | 0.000000000 | 0.000000 | 0 |
| `M-B7_DROPOUT_LONG` | 0.133373 (seed 42) | 0.181818 | 0.886076 | 0.000000000 | -0.000791 | 0 |
| `M-B7_MISSING_FRAME_1PCT` | 0.022130 (seed 42) | 0.050000 | 0.974684 | 0.000000000 | -0.000148 | 0 |
| `M-B7_MISSING_FRAME_5PCT` | 0.022130 (seed 42) | 0.050000 | 0.974684 | 0.000000000 | -0.000989 | 0 |
| `M-B7_MOTION_BURST_MILD` | 0.066343 (seed 42) | 0.108108 | 0.835443 | 0.003037975 | -0.029124 | 0 |
| `M-B7_MOTION_BURST_SEVERE` | 0.178056 (seed 42) | 0.243243 | 0.721519 | 0.014303797 | -0.047172 | 0 |
| `M-B7_COMBINED_MODERATE` | 0.191766 (seed 42) | 0.300000 | 0.670886 | 0.000000000 | -0.003758 | 1 |

## Every seed and profile

| Profile | Seed | Macro F1 | Accuracy | F1 degradation | Max recall degradation | Top-1 vs clean | Mean confidence | Collapse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M-B7_GAUSSIAN_SNR20` | 42 | 0.560091 | 0.632911 | 0.106140 | 0.100000 | 0.886076 | 0.548012 | false |
| `M-B7_GAUSSIAN_SNR20` | 43 | 0.406205 | 0.417722 | 0.035036 | 0.054054 | 0.924051 | 0.350227 | false |
| `M-B7_GAUSSIAN_SNR20` | 44 | 0.306118 | 0.506329 | 0.022989 | 0.045455 | 0.949367 | 0.495896 | false |
| `M-B7_GAUSSIAN_SNR10` | 42 | 0.570543 | 0.632911 | 0.095688 | 0.136363 | 0.797468 | 0.524822 | false |
| `M-B7_GAUSSIAN_SNR10` | 43 | 0.445962 | 0.455696 | 0.000000 | 0.054054 | 0.898734 | 0.353540 | false |
| `M-B7_GAUSSIAN_SNR10` | 44 | 0.367475 | 0.518987 | 0.000000 | 0.054054 | 0.860759 | 0.472211 | false |
| `M-B7_GAUSSIAN_POST_B1_SNR20` | 42 | 0.576505 | 0.632911 | 0.089725 | 0.135135 | 0.848101 | 0.544155 | false |
| `M-B7_GAUSSIAN_POST_B1_SNR20` | 43 | 0.435094 | 0.443038 | 0.006146 | 0.050000 | 0.949367 | 0.349931 | false |
| `M-B7_GAUSSIAN_POST_B1_SNR20` | 44 | 0.374424 | 0.531646 | 0.000000 | 0.027027 | 0.936709 | 0.473447 | false |
| `M-B7_GAUSSIAN_POST_B1_SNR10` | 42 | 0.509443 | 0.544304 | 0.156787 | 0.351351 | 0.645570 | 0.520718 | false |
| `M-B7_GAUSSIAN_POST_B1_SNR10` | 43 | 0.342263 | 0.341772 | 0.098977 | 0.270270 | 0.860759 | 0.352848 | false |
| `M-B7_GAUSSIAN_POST_B1_SNR10` | 44 | 0.427385 | 0.493671 | 0.000000 | 0.567568 | 0.329114 | 0.466723 | false |
| `M-B7_AMP_X0_50` | 42 | 0.244956 | 0.481013 | 0.421275 | 0.500000 | 0.708861 | 0.678995 | true |
| `M-B7_AMP_X0_50` | 43 | 0.515954 | 0.632911 | 0.000000 | 0.500000 | 0.354430 | 0.346420 | false |
| `M-B7_AMP_X0_50` | 44 | 0.214493 | 0.468354 | 0.114614 | 0.136364 | 0.911392 | 0.567346 | true |
| `M-B7_AMP_X0_75` | 42 | 0.489174 | 0.607595 | 0.177057 | 0.350000 | 0.848101 | 0.600969 | false |
| `M-B7_AMP_X0_75` | 43 | 0.541316 | 0.582278 | 0.000000 | 0.300000 | 0.632911 | 0.346123 | false |
| `M-B7_AMP_X0_75` | 44 | 0.216374 | 0.468354 | 0.112732 | 0.136364 | 0.924051 | 0.529420 | true |
| `M-B7_AMP_X1_25` | 42 | 0.563230 | 0.632911 | 0.103001 | 0.227272 | 0.721519 | 0.539260 | false |
| `M-B7_AMP_X1_25` | 43 | 0.377133 | 0.392405 | 0.064107 | 0.135135 | 0.860759 | 0.356210 | false |
| `M-B7_AMP_X1_25` | 44 | 0.453397 | 0.582278 | 0.000000 | 0.000000 | 0.898734 | 0.478441 | false |
| `M-B7_AMP_X1_50` | 42 | 0.515542 | 0.582278 | 0.150688 | 0.272727 | 0.670886 | 0.543710 | false |
| `M-B7_AMP_X1_50` | 43 | 0.353443 | 0.379747 | 0.087797 | 0.243243 | 0.658228 | 0.363627 | false |
| `M-B7_AMP_X1_50` | 44 | 0.511912 | 0.620253 | 0.000000 | 0.000000 | 0.810127 | 0.468157 | false |
| `M-B7_DRIFT_MILD` | 42 | 0.644101 | 0.708861 | 0.022130 | 0.050000 | 0.974684 | 0.554935 | false |
| `M-B7_DRIFT_MILD` | 43 | 0.425612 | 0.443038 | 0.015629 | 0.045454 | 0.974684 | 0.349832 | false |
| `M-B7_DRIFT_MILD` | 44 | 0.328942 | 0.518987 | 0.000164 | 0.000000 | 0.987342 | 0.501236 | false |
| `M-B7_DRIFT_SEVERE` | 42 | 0.641959 | 0.708861 | 0.024272 | 0.045454 | 0.962025 | 0.554836 | false |
| `M-B7_DRIFT_SEVERE` | 43 | 0.459751 | 0.481013 | 0.000000 | 0.000000 | 0.949367 | 0.349832 | false |
| `M-B7_DRIFT_SEVERE` | 44 | 0.328942 | 0.518987 | 0.000164 | 0.000000 | 0.987342 | 0.500346 | false |
| `M-B7_DROPOUT_SHORT` | 42 | 0.666231 | 0.721519 | 0.000000 | 0.000000 | 1.000000 | 0.556369 | false |
| `M-B7_DROPOUT_SHORT` | 43 | 0.451053 | 0.468354 | 0.000000 | 0.000000 | 0.987342 | 0.349782 | false |
| `M-B7_DROPOUT_SHORT` | 44 | 0.351165 | 0.531646 | 0.000000 | 0.000000 | 0.987342 | 0.502275 | false |
| `M-B7_DROPOUT_LONG` | 42 | 0.532858 | 0.632911 | 0.133373 | 0.181818 | 0.898734 | 0.566357 | false |
| `M-B7_DROPOUT_LONG` | 43 | 0.457595 | 0.493671 | 0.000000 | 0.050000 | 0.886076 | 0.348991 | false |
| `M-B7_DROPOUT_LONG` | 44 | 0.305371 | 0.506329 | 0.023736 | 0.045455 | 0.974684 | 0.509296 | false |
| `M-B7_MISSING_FRAME_1PCT` | 42 | 0.644101 | 0.708861 | 0.022130 | 0.050000 | 0.974684 | 0.555429 | false |
| `M-B7_MISSING_FRAME_1PCT` | 43 | 0.451053 | 0.468354 | 0.000000 | 0.000000 | 0.987342 | 0.349931 | false |
| `M-B7_MISSING_FRAME_1PCT` | 44 | 0.329107 | 0.518987 | 0.000000 | 0.000000 | 1.000000 | 0.501978 | false |
| `M-B7_MISSING_FRAME_5PCT` | 42 | 0.644101 | 0.708861 | 0.022130 | 0.050000 | 0.974684 | 0.554688 | false |
| `M-B7_MISSING_FRAME_5PCT` | 43 | 0.451053 | 0.468354 | 0.000000 | 0.000000 | 0.987342 | 0.349832 | false |
| `M-B7_MISSING_FRAME_5PCT` | 44 | 0.329107 | 0.518987 | 0.000000 | 0.000000 | 1.000000 | 0.501137 | false |
| `M-B7_MOTION_BURST_MILD` | 42 | 0.599888 | 0.658228 | 0.066343 | 0.108108 | 0.835443 | 0.531201 | false |
| `M-B7_MOTION_BURST_MILD` | 43 | 0.413657 | 0.430380 | 0.027584 | 0.081081 | 0.886076 | 0.353540 | false |
| `M-B7_MOTION_BURST_MILD` | 44 | 0.398853 | 0.556962 | 0.000000 | 0.027027 | 0.886076 | 0.473002 | false |
| `M-B7_MOTION_BURST_SEVERE` | 42 | 0.488175 | 0.544304 | 0.178056 | 0.243243 | 0.746835 | 0.528728 | false |
| `M-B7_MOTION_BURST_SEVERE` | 43 | 0.360961 | 0.379747 | 0.080279 | 0.189189 | 0.759494 | 0.358436 | false |
| `M-B7_MOTION_BURST_SEVERE` | 44 | 0.514449 | 0.620253 | 0.000000 | 0.081081 | 0.721519 | 0.454955 | false |
| `M-B7_COMBINED_MODERATE` | 42 | 0.474464 | 0.594937 | 0.191766 | 0.300000 | 0.835443 | 0.588805 | false |
| `M-B7_COMBINED_MODERATE` | 43 | 0.561493 | 0.594937 | 0.000000 | 0.200000 | 0.670886 | 0.346025 | false |
| `M-B7_COMBINED_MODERATE` | 44 | 0.271861 | 0.493671 | 0.057245 | 0.050000 | 0.974684 | 0.522943 | true |

Softmax confidence is reported only as the maximum dequantized output value; it is
not interpreted as a calibrated probability.

## Perturbation fidelity

| Profile | Independently regenerable magnitude evidence | Replay identical |
|---|---|---:|
| `M-B7_GAUSSIAN_SNR20` | achieved SNR mean 20.043140 dB | True |
| `M-B7_GAUSSIAN_SNR10` | achieved SNR mean 10.005545 dB | True |
| `M-B7_GAUSSIAN_POST_B1_SNR20` | achieved SNR mean 20.019329 dB | True |
| `M-B7_GAUSSIAN_POST_B1_SNR10` | achieved SNR mean 9.993674 dB | True |
| `M-B7_AMP_X0_50` | scale 0.50; max formula error 0.0 | True |
| `M-B7_AMP_X0_75` | scale 0.75; max formula error 0.0 | True |
| `M-B7_AMP_X1_25` | scale 1.25; max formula error 0.0 | True |
| `M-B7_AMP_X1_50` | scale 1.50; max formula error 0.0 | True |
| `M-B7_DRIFT_MILD` | 0.05 Hz; amplitude multiplier 0.25 | True |
| `M-B7_DRIFT_SEVERE` | 0.05 Hz; amplitude multiplier 0.50 | True |
| `M-B7_DROPOUT_SHORT` | 5 samples; exact masks=True | True |
| `M-B7_DROPOUT_LONG` | 30 samples; exact masks=True | True |
| `M-B7_MISSING_FRAME_1PCT` | 3 removed; rejected=0 | True |
| `M-B7_MISSING_FRAME_5PCT` | 15 removed; rejected=0 | True |
| `M-B7_MOTION_BURST_MILD` | 5 samples; deterministic | True |
| `M-B7_MOTION_BURST_SEVERE` | 5 samples; deterministic | True |
| `M-B7_COMBINED_MODERATE` | Gaussian achieved SNR mean 20.080890 dB | True |

## Frozen preprocessing attenuation

| Profile | Injection domain | Mean pre-B1 delta RMS | Mean post-B1 delta RMS | Mean post/pre ratio |
|---|---|---:|---:|---:|
| `M-B7_GAUSSIAN_SNR20` | `CANONICAL_PHASE_PRE_B1` | 1.381208458 | 0.156302041 | 0.111375064 |
| `M-B7_GAUSSIAN_SNR10` | `CANONICAL_PHASE_PRE_B1` | 4.336344776 | 0.483290393 | 0.113347713 |
| `M-B7_GAUSSIAN_POST_B1_SNR20` | `POST_B1_MODEL_INPUT` | N/A | 0.091294538 | N/A |
| `M-B7_GAUSSIAN_POST_B1_SNR10` | `POST_B1_MODEL_INPUT` | N/A | 0.289622515 | N/A |
| `M-B7_AMP_X0_50` | `CANONICAL_PHASE_PRE_B1` | 6.906615362 | 0.4563665 | 0.106623947 |
| `M-B7_AMP_X0_75` | `CANONICAL_PHASE_PRE_B1` | 3.453307681 | 0.22818325 | 0.106623947 |
| `M-B7_AMP_X1_25` | `CANONICAL_PHASE_PRE_B1` | 3.453307681 | 0.22818325 | 0.106623947 |
| `M-B7_AMP_X1_50` | `CANONICAL_PHASE_PRE_B1` | 6.906615362 | 0.4563665 | 0.106623947 |
| `M-B7_DRIFT_MILD` | `CANONICAL_PHASE_PRE_B1` | 2.441857279 | 0.017217455 | 0.007014041 |
| `M-B7_DRIFT_SEVERE` | `CANONICAL_PHASE_PRE_B1` | 4.883714558 | 0.034358281 | 0.006673247 |
| `M-B7_DROPOUT_SHORT` | `CANONICAL_PHASE_PRE_B1` | 0.089888332 | 0.013522903 | 0.144775836 |
| `M-B7_DROPOUT_LONG` | `CANONICAL_PHASE_PRE_B1` | 0.922126231 | 0.226385586 | 0.240898291 |
| `M-B7_MISSING_FRAME_1PCT` | `CANONICAL_PHASE_PRE_B1` | 0.036287083 | 0.003291887 | 0.093176911 |
| `M-B7_MISSING_FRAME_5PCT` | `CANONICAL_PHASE_PRE_B1` | 0.086292473 | 0.00823488 | 0.094223796 |
| `M-B7_MOTION_BURST_MILD` | `CANONICAL_PHASE_PRE_B1` | 2.544764278 | 0.505031556 | 0.200102447 |
| `M-B7_MOTION_BURST_SEVERE` | `CANONICAL_PHASE_PRE_B1` | 5.089528556 | 1.03468829 | 0.202951611 |
| `M-B7_COMBINED_MODERATE` | `CANONICAL_PHASE_PRE_B1` | 3.617166323 | 0.265065149 | 0.108985985 |

Pre-B1 amplitude and drift behavior must not be interpreted as model invariance when
the frozen BPF/Z-score stage attenuates the injected signal. Post-B1 Gaussian profiles
separately probe model-input robustness.

## Subject-level and class findings

All 17 fixed VALIDATION subjects are retained in `subject_level_robustness.json`, with
per-class TP/FP/TN/FN, precision, recall, F1, prediction distribution, and clean deltas
for every seed/profile. This is one fixed subject set, not subject-split cross-validation.

New collapse conditions: [{"profile_id": "M-B7_AMP_X0_50", "affected_seeds": [42, 44]}, {"profile_id": "M-B7_AMP_X0_75", "affected_seeds": [44]}, {"profile_id": "M-B7_COMBINED_MODERATE", "affected_seeds": [44]}]

## Fallback recommendations

| Profile | Recommendation | Invalid/fallback samples |
|---|---|---:|
| `M-B7_GAUSSIAN_SNR20` | `INFERENCE_VALID_FOR_OFFLINE_STRESS_EVALUATION` | 0 |
| `M-B7_GAUSSIAN_SNR10` | `INFERENCE_VALID_FOR_OFFLINE_STRESS_EVALUATION` | 0 |
| `M-B7_GAUSSIAN_POST_B1_SNR20` | `INFERENCE_VALID_FOR_OFFLINE_STRESS_EVALUATION` | 0 |
| `M-B7_GAUSSIAN_POST_B1_SNR10` | `INFERENCE_VALID_FOR_OFFLINE_STRESS_EVALUATION` | 0 |
| `M-B7_AMP_X0_50` | `INFERENCE_VALID_FOR_OFFLINE_STRESS_EVALUATION` | 0 |
| `M-B7_AMP_X0_75` | `INFERENCE_VALID_FOR_OFFLINE_STRESS_EVALUATION` | 0 |
| `M-B7_AMP_X1_25` | `INFERENCE_VALID_FOR_OFFLINE_STRESS_EVALUATION` | 0 |
| `M-B7_AMP_X1_50` | `INFERENCE_VALID_FOR_OFFLINE_STRESS_EVALUATION` | 0 |
| `M-B7_DRIFT_MILD` | `INFERENCE_VALID_FOR_OFFLINE_STRESS_EVALUATION` | 0 |
| `M-B7_DRIFT_SEVERE` | `INFERENCE_VALID_FOR_OFFLINE_STRESS_EVALUATION` | 0 |
| `M-B7_DROPOUT_SHORT` | `INFERENCE_VALID_FOR_OFFLINE_STRESS_EVALUATION` | 0 |
| `M-B7_DROPOUT_LONG` | `INFERENCE_VALID_FOR_OFFLINE_STRESS_EVALUATION` | 0 |
| `M-B7_MISSING_FRAME_1PCT` | `INFERENCE_VALID_FOR_OFFLINE_STRESS_EVALUATION` | 0 |
| `M-B7_MISSING_FRAME_5PCT` | `INFERENCE_VALID_FOR_OFFLINE_STRESS_EVALUATION` | 0 |
| `M-B7_MOTION_BURST_MILD` | `INFERENCE_VALID_FOR_OFFLINE_STRESS_EVALUATION` | 0 |
| `M-B7_MOTION_BURST_SEVERE` | `INFERENCE_VALID_FOR_OFFLINE_STRESS_EVALUATION` | 0 |
| `M-B7_COMBINED_MODERATE` | `INFERENCE_VALID_FOR_OFFLINE_STRESS_EVALUATION` | 0 |

These are recommendations for later integration work only; M-B7 does not change a
runtime or risk policy.

## Worst conditions

- Macro-F1 degradation: `M-B7_AMP_X0_50`
- Per-class recall degradation: `M-B7_GAUSSIAN_POST_B1_SNR10`
- Top-1 agreement: `M-B7_GAUSSIAN_POST_B1_SNR10`
- Saturation: `M-B7_MOTION_BURST_SEVERE`
- Confidence degradation: `M-B7_MOTION_BURST_SEVERE`

## Limitations and claim boundary

This experiment injects deterministic synthetic perturbations into real, canonical
VALIDATION windows. It does not measure MR60 hardware, Raspberry Pi execution, a live
acquisition path, deployment readiness, or clinical apnea. SafeNest `APNEA` remains a
voluntary breath-hold proxy. Timestamp jitter was not added because the fixed matrix did
not preregister a magnitude; missing-frame damage does use the approved A3 timeline and
resampling contract.
