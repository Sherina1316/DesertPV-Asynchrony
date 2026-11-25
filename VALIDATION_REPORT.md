# v24.22 validation report

Validation completed in the packaged source tree:

- Python compilation: passed
- Shell syntax validation: passed
- Complete regression suite: 105 passed, 0 failed
- Existing far-remote new-site recovery tests: passed
- Existing stable-dark rejection tests: passed
- New true-2025 remote-site novelty test: passed
- New long-standing dark parcel with small 2025 noise rejection test: passed
- New persistent high-model false-positive rejection test: passed

The new tests explicitly verify that:

1. a far remote 2025 PV site remains detectable without adjacency or annual candidate support;
2. a parcel dark for several years is rejected even if 2025 is slightly darker;
3. a persistent high-model-probability non-PV object cannot enter through either the strict remote or full-domain remote path;
4. weak object edges may be recovered only from genuine terminal-novelty seeds.

These are software and synthetic-contract tests. They do not substitute for independent map accuracy validation on the 142 real units.
