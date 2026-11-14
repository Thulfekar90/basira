# Basira in a container:
#   docker build -t basira .
#   docker run --rm --network none --user "$(id -u):$(id -g)" -v "$PWD:/data" basira scan photo.jpg
#
# UNTESTED: this file was written without a Docker daemon at hand and has not been built
# or run by the project. It is kept deliberately simple; please report what breaks.
#
# - Build needs the network once (pip downloads the dependencies).
# - Running needs none: start it with `--network none`. Basira has no network code in its
#   analysis path; the flag makes that a property of the container rather than a promise.
# - Runs as an unprivileged user. Mount the files to analyse at /data; add
#   `--user "$(id -u):$(id -g)"` so that reports are written with your own ownership.
#   With a read-only mount (`-v "$PWD:/data:ro"`) add `--no-html` or
#   `--report-dir /tmp/reports`, because reports are written next to the files by default.
# - `basira serve` binds to 127.0.0.1 inside the container and is therefore not reachable
#   from the host; use `scan` here.

FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY basira ./basira
RUN pip install . && rm -rf /src/build /src/*.egg-info

RUN useradd --create-home --uid 10001 basira
USER basira
WORKDIR /data

# Fails the build if the installation is incomplete (libraries, ffmpeg, self-test).
RUN basira doctor

ENTRYPOINT ["basira"]
CMD ["--help"]
