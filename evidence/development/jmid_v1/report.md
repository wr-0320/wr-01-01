# JMID v1 exploratory development result

This is three-seed exploratory evidence, not formal confirmation.

## Condition means

| Condition | Method | Accuracy | Brier | NLL | ECE | Wrong confidence | High-confidence error |
|---|---|---:|---:|---:|---:|---:|---:|
| A_random_independent | M1_naive | 0.8761 | 0.0859 | 0.2818 | 0.0089 | 0.7042 | 0.0280 |
| A_random_independent | M2_missing | 0.8767 | 0.0859 | 0.2821 | 0.0097 | 0.7072 | 0.0270 |
| A_random_independent | M3_dependence | 0.8764 | 0.0859 | 0.2819 | 0.0102 | 0.7096 | 0.0262 |
| A_random_independent | M4_joint | 0.8761 | 0.0864 | 0.2835 | 0.0138 | 0.7118 | 0.0244 |
| B_strategic_independent | M1_naive | 0.8737 | 0.0875 | 0.2848 | 0.0250 | 0.7027 | 0.0269 |
| B_strategic_independent | M2_missing | 0.9169 | 0.0652 | 0.2195 | 0.0053 | 0.7831 | 0.0152 |
| B_strategic_independent | M3_dependence | 0.8739 | 0.0875 | 0.2850 | 0.0267 | 0.7056 | 0.0247 |
| B_strategic_independent | M4_joint | 0.9169 | 0.0653 | 0.2197 | 0.0071 | 0.7813 | 0.0152 |
| C_random_dependent | M1_naive | 0.8497 | 0.1074 | 0.3532 | 0.0382 | 0.7469 | 0.0403 |
| C_random_dependent | M2_missing | 0.8503 | 0.1074 | 0.3533 | 0.0385 | 0.7492 | 0.0399 |
| C_random_dependent | M3_dependence | 0.8544 | 0.1037 | 0.3350 | 0.0123 | 0.7175 | 0.0184 |
| C_random_dependent | M4_joint | 0.8535 | 0.1042 | 0.3364 | 0.0135 | 0.7195 | 0.0189 |
| D_strategic_dependent | M1_naive | 0.8297 | 0.1141 | 0.3717 | 0.0572 | 0.7139 | 0.0549 |
| D_strategic_dependent | M2_missing | 0.8715 | 0.0980 | 0.3357 | 0.0455 | 0.8046 | 0.0359 |
| D_strategic_dependent | M3_dependence | 0.8309 | 0.1111 | 0.3531 | 0.0254 | 0.6776 | 0.0149 |
| D_strategic_dependent | M4_joint | 0.8714 | 0.0933 | 0.3051 | 0.0111 | 0.7253 | 0.0221 |

## Condition D generator check

- missing_rate: 0.2213
- missing_rate_z0: 0.1228
- missing_rate_z1: 0.3169
- pair_correlation: 0.4214
- missing_pair_correlation: 0.7669

## Shift endpoints

### Joint strength 0.75
- M1_naive: Brier 0.1079; NLL 0.3542
- M2_missing: Brier 0.1011; NLL 0.3341
- M3_dependence: Brier 0.1076; NLL 0.3519
- M4_joint: Brier 0.1008; NLL 0.3324

### Joint strength 1.0
- M1_naive: Brier 0.1172; NLL 0.3878
- M2_missing: Brier 0.1075; NLL 0.3583
- M3_dependence: Brier 0.1170; NLL 0.3849
- M4_joint: Brier 0.1066; NLL 0.3539
