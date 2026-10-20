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

## Bounded retry on 2026-10-20

The current runtime was checked once after the audit was frozen. It still
reported source configuration `cecfg_6ad5ceafc7708190b3adb049e20be9e2`
(`spec_revision=1`). None of the nine saved scholarly hosts appeared in
`effective_allowed_hosts`, so no request was repeated against those known-
blocked destinations.

A local search found no full text for any of the 15 frozen works. A single
fallback search through the allowed public GitHub web host returned an HTTP 200
page, but GitHub required authentication before exposing code-search results;
it did not yield a paper body. This metadata/search page is not counted as
full-text access.

The bounded access retry is therefore exhausted for this runtime. The saved
draft must be published and loaded by a new instance before one final access
check. If that instance still cannot retrieve sufficient lawful full texts,
the audit remains `INCONCLUSIVE-BLOCKED`; the protocol, matrix and literature
set must not be altered to manufacture a direction decision.
