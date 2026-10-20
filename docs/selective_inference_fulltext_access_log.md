# Selective-inference full-text access log

Date: 2026-10-20

## Current runtime

The managed environment reports an enforced restricted network policy. Its
effective host list contains package-manager and GitHub destinations but does
not contain arXiv, PMLR, JMLR, OpenReview, IEEE Xplore, ACM Digital Library or
Springer.

No repeated paper request was made against a policy known not to include those
destinations. The literature matrix therefore records `metadata_only`; no row
is treated as full-text verified.

## Saved configuration change

A network configuration draft now includes:

- `arxiv.org` and `export.arxiv.org`;
- `proceedings.mlr.press` and `jmlr.org`;
- `openreview.net`;
- `ieeexplore.ieee.org`, `dl.acm.org`, `link.springer.com` and
  `academic.oup.com`.

The draft requires review and publication. Saving it does not modify the
current instance. After a new instance loads the published policy, access must
again distinguish metadata pages, abstracts and actual paper bodies.

Until the frozen primary works are read in full, the direction decision remains
`INCONCLUSIVE-BLOCKED` for originality and `NO-GO` for implementation.
