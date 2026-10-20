# Route B scholarly full-text connectivity check

Date: 2026-10-20  
Scope: **network validation only**  
Outcome: **published rules are not active in the current runtime; all tested
scholarly paths remain blocked by the managed proxy**

## Protocol controls

Before testing, the frozen audit protocol, evidence matrix, access log,
direction decision and GO/NO-GO protocol were read. This check did not modify
the evidence matrix, review any theorem, change the research thresholds, or
start algorithm, theorem or simulator development.

## Runtime-policy observation

The runtime environment-status endpoint twice returned `The environment
configuration changed. Try again.` The effective local policy snapshot at
`/etc/codex/network-policy.json` remained a restricted policy whose egress list
did not contain `arxiv.org`, `ieeexplore.ieee.org`, `link.springer.com` or
`dl.acm.org`. Therefore the published draft has not been applied to this running
instance, even though it may have been accepted for a replacement environment.

No network rule was changed during this check.

## Results

Each request followed redirects, used a 20-second limit, saved any response body
under `/tmp`, and tested both the reported content type and the `%PDF-` file
signature. `HTTP 000` means the HTTPS tunnel failed before an origin-server HTTP
response was received; the reported `403` is the managed proxy's CONNECT
response.

| Source | Access class | Requested and final URL | Result | Actual content obtained |
|---|---|---|---|---|
| arXiv | metadata/abstract page | `https://arxiv.org/abs/0710.3742` | curl 56; HTTP 000; CONNECT 403 | none |
| arXiv | PDF | `https://arxiv.org/pdf/0710.3742` | curl 56; HTTP 000; CONNECT 403 | no PDF |
| IEEE Xplore | metadata/abstract page | `https://ieeexplore.ieee.org/document/6587660` | curl 56; HTTP 000; CONNECT 403 | none |
| IEEE Xplore | document/PDF path | `https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6587660` | curl 56; HTTP 000; CONNECT 403 | no PDF |
| SpringerLink | book metadata page | `https://link.springer.com/book/10.1007/978-3-540-35653-0` | curl 56; HTTP 000; CONNECT 403 | none |
| SpringerLink | PDF path | `https://link.springer.com/content/pdf/10.1007/978-3-540-35653-0.pdf` | curl 56; HTTP 000; CONNECT 403 | no PDF |
| ACM Digital Library | metadata/abstract page | `https://dl.acm.org/doi/10.1145/945445.945450` | curl 56; HTTP 000; CONNECT 403 | none |
| ACM Digital Library | PDF path | `https://dl.acm.org/doi/pdf/10.1145/945445.945450` | curl 56; HTTP 000; CONNECT 403 | no PDF |

The final URL for every request was unchanged because the proxy rejected the
CONNECT tunnel before redirects or origin content were available. Content type,
response bytes and PDF signature were absent in every case.

## Stage-one decision

No tested source currently provides even a metadata page in this runtime, so no
target paper full text is obtainable here. The literature verdict remains
`INCONCLUSIVE-BLOCKED`, and the implementation decision remains `NO-GO`.

Do not repeat these same requests in this instance. Resume connectivity testing
only after a replacement runtime reports the academic hosts in its effective
policy, or after lawful full texts are supplied locally. The frozen evidence
matrix must remain unchanged until an original paper body can be verified.

