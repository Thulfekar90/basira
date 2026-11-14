"""Worker functions for the process-pool tests. They live in their own module because
spawned workers must be able to import them."""
import os
import subprocess
import sys

MARK = "61.25"          # an unusual sleep length, so the test can find exactly its process


def start_child_and_hang(item, opts):
    subprocess.run([sys.executable, "-c", f"import time; time.sleep({MARK})"])
    return {"index": item["index"], "path": item["path"]}


def start_child_and_crash(item, opts):
    subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({MARK})"])
    os._exit(7)
