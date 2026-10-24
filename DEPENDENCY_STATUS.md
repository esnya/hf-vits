# Security dependency patch (2026-10-24)

This patch updates only urllib3 2.7.0 -> 2.8.0 and development GitPython
3.1.59 -> 3.2.1, with hashes from their exact public PyPI releases. All other
lock records, Pipfile, package metadata and the documented Python 3.10 contract
are preserved. It is an offline targeted lock edit, not a resolver regeneration.
The pre-existing Pipfile hash is stale; `pipenv verify` still reports it as
out-of-date. This patch does not replace that metadata or imply a full sync.

Local reviewed high/critical advisory matching falls from six distinct
package/advisory pairs to one. The remaining match is filelock 3.32.5,
[GHSA-v4x8-3wr5-3wq8](https://github.com/advisories/GHSA-v4x8-3wr5-3wq8).
Its minimum fix, filelock 4.2.1, declares Python >=3.11. Updating it would change
the explicitly documented runtime; that migration is not applied here.

The advisory affects SQLite-backed ReadWriteLock/AsyncReadWriteLock with a
different local user able to alter the lock directory. No application use of
those classes was found. Static inspection of the exact, hash-verified
Hugging Face Hub 1.29.0 wheel finds WeakFileLock using FileLock/SoftFileLock,
with no ReadWriteLock reference. This narrows the observed path; it is not an
upstream fix or proof about every dependency/runtime path.

Three public-package native checks pass on Python 3.10: exact patched package
versions; rejection of a 70KB chunk-size line by the maximum-length guard; and GitPython
author parsing/local Git repository operations. Both patched dependency edges
fit the existing Requests/GitDB constraints. No existing test suite or CI
workflow is present. Full package installation, model loading and inference
are not validated by these checks.

A full environment rebuild needs a separate review: Pipfile selects Torch's
CUDA 11.8 source, while the existing lock selects PyPI Torch 2.13.0 and contains
CUDA 12/13 dependencies. That baseline mismatch is not changed by this patch.
The next unapplied security candidate is Python 3.11 plus filelock 4.2.1, then a
coherent Torch source/version and VITS tokenizer/checkpoint compatibility
verification. Do not treat this partial patch as a complete runtime migration.

No model/GPU downloads, license acceptance, source build hooks, deployment or
external tree audit were used. Primary metadata:
[urllib3 2.8.0](https://pypi.org/pypi/urllib3/2.8.0/json),
[GitPython 3.2.1](https://pypi.org/pypi/GitPython/3.2.1/json),
[filelock 4.2.1](https://pypi.org/pypi/filelock/4.2.1/json).
