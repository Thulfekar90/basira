"""Check what `python -m build` produced before anyone installs or uploads it.

    python .github/scripts/check_dist.py dist

* the wheel contains the `basira` package and its metadata, nothing else
  (no tests, benchmarks, examples, datasets or virtual environments);
* the sdist contains the licence, both READMEs, the docs and the tests, and none of the
  generated output;
* both carry the version that basira/__init__.py declares.
"""
import glob
import os
import re
import sys
import tarfile
import zipfile


def fail(msg):
    print("check_dist: FAIL:", msg)
    sys.exit(1)


def main(dist):
    wheels = glob.glob(os.path.join(dist, "basira-*.whl"))
    sdists = glob.glob(os.path.join(dist, "basira-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        fail(f"expected one wheel and one sdist in {dist}, found {wheels} and {sdists}")

    with zipfile.ZipFile(wheels[0]) as z:
        names = z.namelist()
        meta = z.read(next(n for n in names if n.endswith(".dist-info/METADATA"))).decode()
        init = z.read("basira/__init__.py").decode()
    version = re.search(r'^__version__ = "([^"]+)"', init, re.M).group(1)
    tops = {n.split("/")[0] for n in names}
    if tops != {"basira", f"basira-{version}.dist-info"}:
        fail(f"unexpected top-level entries in the wheel: {sorted(tops)}")
    if not all(n.endswith(".py") for n in names if n.startswith("basira/")):
        fail("non-Python files inside the package: "
             + str([n for n in names if n.startswith("basira/") and not n.endswith(".py")]))
    if f"\nVersion: {version}\n" not in meta:
        fail(f"wheel metadata does not say Version: {version}")
    if "basira/cli.py" not in names or not any(n.endswith("entry_points.txt") for n in names):
        fail("the command-line entry point is missing from the wheel")
    if not any("LICENSE" in n for n in names):
        fail("the licence file is missing from the wheel")
    size = os.path.getsize(wheels[0])
    if size > 2_000_000:
        fail(f"the wheel is {size} bytes; something large was packaged by mistake")

    with tarfile.open(sdists[0]) as t:
        members = [m.name.split("/", 1)[1] for m in t.getmembers() if "/" in m.name]
    for needed in ("LICENSE", "README.md", "README.ar.md", "pyproject.toml", "CHANGELOG.md",
                   "docs/METHODS.md", "docs/USAGE.md", "docs/JSON.md", "tests/conftest.py",
                   "tests/synth_image.py", "basira/cli.py"):
        if needed not in members:
            fail(f"{needed} is missing from the sdist")
    for m in members:
        if m.startswith(("examples/out", "build/", ".venv", "dist/", ".git")) \
                or m.endswith((".basira.html", ".pyc")) or "/__pycache__" in m:
            fail(f"{m} must not be in the sdist")
    print(f"check_dist: ok - {os.path.basename(wheels[0])} ({size} bytes, {len(names)} files), "
          f"{os.path.basename(sdists[0])} ({len(members)} entries), version {version}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dist")
