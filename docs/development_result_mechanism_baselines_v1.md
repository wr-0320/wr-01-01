# Mechanism baseline development result v1

## Status and design

This is three-seed development evidence used to choose a formal confirmation
protocol. It is not a formal result. Seeds 9101--9103, seven intelligence
families, 100 paired scenarios per family and method, and five methods produced
10,500 model-episode evaluations.

All methods share the fitting data, mechanism-label availability, reliable
Gaussian location and scale estimates, validation cases, fixed inference time,
temperature-selection rule, and paired test scenarios. The global dual mixture
uses the labeled compromised component. The other robust baselines have access
to the same labels but omit that component by design:

- single mechanism: reliable Gaussian likelihood;
- global dual mixture: reliable/compromised Gaussian mixture with one fitted
  global mechanism prior;
- Student-t: reliable Student-t likelihood, with degrees of freedom selected on
  validation NLL from `[3, 5, 10, 30]`;
- Huber: standardized-residual Huber pseudo-likelihood, with delta selected from
  `[1, 1.345, 2, 3]`;
- reliability weighted: reliable Gaussian evidence multiplied linearly by the
  fitted source-region reliability, using the fixed interpretable weight `w=ρ`.

Every candidate receives independent temperature calibration on the same
validation cases. No test result is used for selection.

## Results

| Method | Accuracy | Brier | NLL | ECE |
|---|---:|---:|---:|---:|
| Single mechanism | 77.57% | 0.2672 | 0.4235 | 0.0543 |
| Global dual mixture | 81.52% | 0.1993 | 0.2873 | 0.0452 |
| Student-t | 80.62% | 0.2149 | 0.3208 | 0.0420 |
| Huber | 81.05% | 0.2129 | 0.3195 | 0.0420 |
| Reliability weighted | 77.48% | 0.2666 | 0.4220 | 0.0534 |

Global dual mixture minus each comparator:

| Comparator | Accuracy difference (pp, 95% CI) | Brier | NLL |
|---|---:|---:|---:|
| Single mechanism | +3.95 [-1.47, +9.37] | -0.0679 | -0.1362 |
| Student-t | +0.90 [-2.28, +4.09] | -0.0156 | -0.0335 |
| Huber | +0.48 [-2.39, +3.34] | -0.0136 | -0.0322 |
| Reliability weighted | +4.05 [-1.22, +9.32] | -0.0673 | -0.1347 |

The expanded Student-t and Huber grids selected no upper-bound value; their
reported results are unchanged from the initial narrower-grid check. The Brier
and NLL seed-level intervals against the single-mechanism and
reliability-weighted baselines exclude zero. The intervals against Student-t and
Huber cross zero. Accuracy intervals cross zero for every comparison. None of
these development comparisons establishes a formal claim.

## Interpretation and formal comparator

The global dual mixture remains the method candidate. Huber is the primary
formal robust comparator because it has the strongest development accuracy and
NLL among the non-mixture robust models. Student-t remains a required secondary
comparator. Simple linear reliability weighting does not reproduce the mixture
result.

The formal claim will focus on probability quality. It will require lower NLL
and Brier than Huber together with an accuracy noninferiority guard. Fresh seeds,
an exact run size, and failure rules are frozen in
`docs/formal_confirmation_protocol_v1.md` before the formal command is run.
