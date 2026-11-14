"""The packaging metadata must agree with the code, or a release ships the wrong thing."""
from __future__ import annotations

import importlib
import os
import re

import pytest

import basira

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pyproject() -> dict:
    try:
        import tomllib as toml                # Python 3.11+
    except ImportError:
        toml = pytest.importorskip("tomli")   # older Pythons: pip install -e ".[dev]"
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        return toml.load(f)


def _read(name: str) -> str:
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


def test_version_is_the_same_everywhere():
    project = _pyproject()["project"]
    assert project["version"] == basira.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+([a-z]+\d+)?", basira.__version__)
    # the citation file and the changelog name the same version
    assert re.search(r"^version: \"?%s\"?$" % re.escape(basira.__version__),
                     _read("CITATION.cff"), re.M)
    assert f"[{basira.__version__}]" in _read("CHANGELOG.md")


def test_console_script_resolves_to_a_callable():
    scripts = _pyproject()["project"]["scripts"]
    assert set(scripts) == {"basira"}
    module, _, attr = scripts["basira"].partition(":")
    func = getattr(importlib.import_module(module), attr)
    assert callable(func)
    with pytest.raises(SystemExit) as e:          # and it is the real CLI
        func(["--version"])
    assert e.value.code == 0


def test_installed_entry_point_matches_if_installed():
    try:
        from importlib.metadata import PackageNotFoundError, distribution
    except ImportError:                                   # pragma: no cover
        pytest.skip("importlib.metadata not available")
    try:
        dist = distribution("basira")
    except PackageNotFoundError:
        pytest.skip("basira is not installed (running from the source tree)")
    eps = [ep for ep in dist.entry_points if ep.group == "console_scripts"]
    assert [(ep.name, ep.value) for ep in eps] == [("basira", "basira.cli:main")]
    assert callable(eps[0].load())


def test_metadata_matches_what_is_supported_and_documented():
    project = _pyproject()["project"]
    assert project["requires-python"] == ">=3.9"
    for minor in (9, 10, 11, 12, 13):
        assert f"Programming Language :: Python :: 3.{minor}" in project["classifiers"]
    assert project["license"] == "MIT" and "MIT License" in _read("LICENSE")
    assert not any(c.startswith("License ::") for c in project["classifiers"]), \
        "an SPDX licence expression and a licence classifier cannot be combined"
    # every runtime dependency is one the README names, and nothing heavy sneaks in
    names = sorted(re.split(r"[<>=!~ ;\[]", d, 1)[0].lower() for d in project["dependencies"])
    assert names == ["imageio-ffmpeg", "numpy", "opencv-python-headless", "pillow", "scipy",
                     "soundfile"]


def test_project_urls_point_at_the_same_repository_as_the_readme():
    """README.md uses github.com/OWNER/basira until the project is published
    (docs/PUBLISHING.md); whoever replaces OWNER must replace it here too."""
    owners = set(re.findall(r"github\.com/([^/\s]+)/basira", _read("README.md")))
    assert len(owners) == 1, owners
    owner = owners.pop()
    urls = _pyproject()["project"]["urls"]
    assert urls and all(u.startswith(f"https://github.com/{owner}/basira") for u in urls.values())
    for name in ("CITATION.cff", "CONTRIBUTING.md", "docs/USAGE.md",
                 ".github/ISSUE_TEMPLATE/config.yml"):
        found = set(re.findall(r"github\.com/([^/\s]+)/basira", _read(name)))
        assert found <= {owner}, (name, found)


def test_only_the_package_is_packaged():
    """The wheel is built from ``basira*`` only: tests, benchmarks and examples must not
    become importable top-level packages."""
    find = _pyproject()["tool"]["setuptools"]["packages"]["find"]
    assert find["include"] == ["basira*"]
    for folder in ("tests", "benchmarks", "examples"):
        assert not os.path.exists(os.path.join(ROOT, folder, "__init__.py"))
    for sub in ("core", "image", "audio", "video"):
        assert os.path.exists(os.path.join(ROOT, "basira", sub, "__init__.py"))


# ----------------------------------------------------------------- documentation stays true

def _python_blocks(markdown: str) -> str:
    return "\n\n".join(re.findall(r"^```python\n(.*?)^```", markdown, re.S | re.M))


def test_the_detector_tutorial_runs(tmp_path):
    """docs/ADDING_A_DETECTOR.md promises that its code works: run it, run its tests, and
    register the example detector the way the page says."""
    import json

    import basira.image as image_engine

    src = _python_blocks(_read("docs/ADDING_A_DETECTOR.md"))
    ns = {"__name__": "adding_a_detector_example"}
    exec(compile(src, "docs/ADDING_A_DETECTOR.md", "exec"), ns)
    tests = [name for name in ns if name.startswith("test_")]
    assert len(tests) >= 3
    for i, name in enumerate(tests):
        folder = tmp_path / str(i)
        folder.mkdir()
        ns[name](folder)

    entry = (ns["DETECTOR"], ns["TITLE"], basira.Category.MANIPULATION, ns["analyze_dupblock"])
    image_engine.PIXEL_DETECTORS.append(entry)
    try:
        path, _ = ns["make_block_clone"](str(tmp_path), 5)
        report = basira.analyze(path)
    finally:
        image_engine.PIXEL_DETECTORS.remove(entry)
    found = report.get("image.dupblock")
    assert found is not None and found.score > 0.5 and found.regions
    assert report.scores["manipulation"] > 0.0
    json.dumps(report.to_dict(), allow_nan=False)
    assert basira.analyze(path).get("image.dupblock") is None      # unregistered again


def test_documented_files_exist():
    """Links between the documents this package ships must not dangle."""
    for name in ("README.md", "README.ar.md", "LICENSE", "CHANGELOG.md", "CONTRIBUTING.md",
                 "CITATION.cff", "docs/USAGE.md", "docs/JSON.md", "docs/METHODS.md",
                 "docs/ADDING_A_DETECTOR.md", "Dockerfile", ".github/workflows/ci.yml"):
        assert os.path.exists(os.path.join(ROOT, name)), name
    for doc in ("CONTRIBUTING.md", "docs/USAGE.md", "docs/JSON.md", "docs/ADDING_A_DETECTOR.md",
                "CHANGELOG.md"):
        base = os.path.dirname(os.path.join(ROOT, doc))
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", _read(doc)):
            if "://" in target or target.startswith("mailto:"):
                continue
            assert os.path.exists(os.path.normpath(os.path.join(base, target))), (doc, target)
