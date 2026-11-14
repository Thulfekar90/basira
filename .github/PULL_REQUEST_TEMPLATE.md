## What this changes

<!-- One topic per pull request. What does it change, and why? -->

## What was measured

<!-- Before / after numbers and the command that produced them. Include anything that got
     WORSE. If a threshold was tuned, say on which data. "Not applicable" is fine for
     changes that cannot affect results (docs, packaging). -->

## Checklist

- [ ] `python -m pytest -q` passes locally.
- [ ] New behaviour has a test; a bug fix has a test that failed before it.
- [ ] Documentation still true: README, `docs/`, `--help` texts, and every command they show.
- [ ] A line under "Unreleased" in `CHANGELOG.md` for anything a user would notice.
- [ ] No new runtime dependency, network access or model download (or explained above).

**If this adds or changes a detector** (see `docs/ADDING_A_DETECTOR.md`):

- [ ] A seeded synthetic forgery **with ground truth** in `tests/synth_*.py`.
- [ ] A **clean-file test**: it stays below its threshold on untouched files.
- [ ] The **threshold is measured**: a named constant with the measurement in a comment.
- [ ] The **blind spots are stated** in `docs/METHODS.md`.
- [ ] Benchmark numbers before and after, on data I did not generate where possible,
      including false positives on clean files.

**If this changes the command line or the JSON output:**

- [ ] Exit statuses and the fields listed as stable in `docs/JSON.md` are unchanged, or
      `schema_version` was raised and the changelog says why.
