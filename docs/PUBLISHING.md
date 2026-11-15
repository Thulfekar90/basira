# Publishing checklist

The repository is ready to push but has never been published. Whoever publishes it
(a person, or an agent such as Codex acting for the owner) should follow this list.

## Brief for an agent

> You are publishing the Python project in this directory (`basira`) to the owner's GitHub
> account. Do not change any code or documentation other than what this brief lists.
>
> 1. Run `python -m venv .venv && .venv/bin/pip install -e ".[dev]" && .venv/bin/python -m pytest -q`.
>    The whole test suite (`.venv/bin/python -m pytest -q`) must pass. If any fail, stop and report; do not publish.
> 2. Ask the owner for their GitHub username and whether the repository should be public.
>    Check that the name `basira` is free on their account. If it is taken, ask for another
>    name; do not pick one yourself.
> 3. Replace the two `OWNER` placeholders in `README.md` and the one in `README.ar.md` with
>    the username. Commit that single change with the message `Set repository owner in install URLs`.
> 4. Create the repository with the description
>    "Explainable forensics for edited and AI-generated images, audio and video: shows where a file was changed and why."
>    and the topics `forensics`, `image-forensics`, `audio-forensics`, `video-forensics`,
>    `deepfake-detection`, `tamper-detection`, `c2pa`, `copy-move`, `python`.
> 5. Push the `main` branch (rename the local branch to `main` if it is `master`). Never
>    force-push. Do not push `.venv/`, `examples/out/` or any file not tracked by git.
> 6. Confirm the GitHub Actions run for the push is green on every required job (the Windows job is experimental and may fail). If it is
>    not, report the failing job and log; do not disable the workflow.
> 7. Report the repository URL and the CI status. Do not create a release, a PyPI upload,
>    or any social-media post: those are the owner's decisions.

## Before announcing it (owner's decisions)

- **Hero image.** The README's first image is a synthetic test scene. A real photograph the
  owner has the rights to (their own, or CC0), edited by hand and analysed with
  `basira scan`, makes a far stronger first impression. Add a screenshot of the HTML report.
- **More public-dataset numbers.** Columbia and CoMoFoD (in-sample) and CASIA v1, CASIA v2
  and COVERAGE (out-of-sample) are in the README. Still missing: IMD2020 (real-world
  edits) for images, and any real data at all for audio (ASVspoof) and video.
- **Name.** Check that `basira` is free on PyPI before promising `pip install basira`.
- **PyPI release** once the above is done: `python -m build && twine upload dist/*`.
