# Route B full-text access log

Date: 2026-10-20  
Outcome: **required primary full texts are unavailable in this cloud instance**

## Local evidence check

A filename search under `/workspace` and `/tmp` found project-generated PDFs,
figures and test outputs. It found no local copies of the primary fault
isolation, secure-estimation, attack-identification or set-membership papers
required by the frozen audit.

## Network check

The running environment reports restricted egress. GitHub and package-manager
hosts are available, while the scholarly full-text hosts needed for this audit
are absent from the effective allowlist. Direct requests to arXiv, IEEE Xplore,
Springer and Semantic Scholar failed at the managed proxy with HTTP CONNECT 403.
An escalated retry to arXiv produced the same proxy denial. GitHub HTML was
reachable, but repository search is neither a scholarly index nor primary full
text.

The access failure establishes only that this environment cannot complete the
audit. It provides no evidence for or against novelty.

## Resumption condition

Resume the matrix only when lawful primary full texts are available locally or
the environment permits the required scholarly hosts. Do not fill theorem
columns from abstracts, snippets, memory or secondary summaries. No further
unchanged connectivity retries are useful before the network policy changes.

