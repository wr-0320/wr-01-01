# JMID v2 falsification result

Three-seed exploratory falsification; not formal confirmation.

## Primary means

| Condition | Method | Accuracy | Brier | NLL | ECE | Wrong conf. | High-conf. error |
|---|---|---:|---:|---:|---:|---:|---:|
| A_random_independent | M2_missing | 0.9075 | 0.0645 | 0.2148 | 0.0080 | 0.7072 | 0.0229 |
| A_random_independent | M3_dependence | 0.9074 | 0.0644 | 0.2148 | 0.0082 | 0.7084 | 0.0224 |
| A_random_independent | M23_fixed | 0.9081 | 0.0645 | 0.2152 | 0.0066 | 0.7115 | 0.0215 |
| A_random_independent | M4_joint | 0.9069 | 0.0647 | 0.2157 | 0.0095 | 0.7104 | 0.0213 |
| B_censor_independent | M2_missing | 0.8948 | 0.0777 | 0.2589 | 0.0127 | 0.7434 | 0.0191 |
| B_censor_independent | M3_dependence | 0.8633 | 0.0960 | 0.3111 | 0.0316 | 0.7248 | 0.0205 |
| B_censor_independent | M23_fixed | 0.8954 | 0.0782 | 0.2601 | 0.0124 | 0.7473 | 0.0189 |
| B_censor_independent | M4_joint | 0.8942 | 0.0781 | 0.2596 | 0.0090 | 0.7368 | 0.0176 |
| C_random_relay | M2_missing | 0.8586 | 0.1119 | 0.3983 | 0.0622 | 0.8240 | 0.0816 |
| C_random_relay | M3_dependence | 0.8623 | 0.1032 | 0.3355 | 0.0098 | 0.7521 | 0.0153 |
| C_random_relay | M23_fixed | 0.8623 | 0.1033 | 0.3358 | 0.0095 | 0.7525 | 0.0151 |
| C_random_relay | M4_joint | 0.8619 | 0.1034 | 0.3359 | 0.0105 | 0.7532 | 0.0149 |
| D_censor_relay | M2_missing | 0.8529 | 0.1149 | 0.3946 | 0.0551 | 0.8076 | 0.0535 |
| D_censor_relay | M3_dependence | 0.7609 | 0.1390 | 0.4353 | 0.1147 | 0.6859 | 0.0341 |
| D_censor_relay | M23_fixed | 0.8323 | 0.1234 | 0.3988 | 0.0865 | 0.7553 | 0.0356 |
| D_censor_relay | M4_joint | 0.8635 | 0.1059 | 0.3421 | 0.0129 | 0.7660 | 0.0126 |

## Condition D diagnostics

- missing_rate: 0.3293
- missing_rate_z0: 0.2306
- missing_rate_z1: 0.4301
- missing_given_raw1: 0.5141
- missing_given_raw0: 0.1481
- copy_rate: 0.8510
- within_state_report_correlation: 0.8511
- within_state_missing_correlation: 0.0917

## Transfer means

### strength_0.75
- M2_missing: Brier 0.1115; NLL 0.3843; accuracy 0.8549
- M3_dependence: Brier 0.1118; NLL 0.3613; accuracy 0.8595
- M23_fixed: Brier 0.1073; NLL 0.3510; accuracy 0.8597
- M4_joint: Brier 0.1045; NLL 0.3414; accuracy 0.8640

### strength_1.0
- M2_missing: Brier 0.1251; NLL 0.4361; accuracy 0.8439
- M3_dependence: Brier 0.1256; NLL 0.4004; accuracy 0.8445
- M23_fixed: Brier 0.1187; NLL 0.3865; accuracy 0.8515
- M4_joint: Brier 0.1152; NLL 0.3724; accuracy 0.8542

### missing_negative
- M2_missing: Brier 0.2171; NLL 0.7632; accuracy 0.7245
- M3_dependence: Brier 0.1797; NLL 0.5602; accuracy 0.7199
- M23_fixed: Brier 0.2319; NLL 0.7275; accuracy 0.6721
- M4_joint: Brier 0.1673; NLL 0.5193; accuracy 0.7643

### missing_conflict
- M2_missing: Brier 0.1244; NLL 0.4340; accuracy 0.8465
- M3_dependence: Brier 0.1565; NLL 0.4931; accuracy 0.7252
- M23_fixed: Brier 0.1266; NLL 0.4040; accuracy 0.8308
- M4_joint: Brier 0.1115; NLL 0.3551; accuracy 0.8535

### rotated_dependency
- M2_missing: Brier 0.1210; NLL 0.4132; accuracy 0.8425
- M3_dependence: Brier 0.1394; NLL 0.4407; accuracy 0.8029
- M23_fixed: Brier 0.1247; NLL 0.4023; accuracy 0.8327
- M4_joint: Brier 0.1332; NLL 0.4174; accuracy 0.8068
