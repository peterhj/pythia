#!/usr/bin/env python3

from argparse import ArgumentParser
import os
import shlex
import stat

def cyan(s: str, bold=False) -> str:
    if bold:
        return f"\x1b[36;1m{s}\x1b[0m"
    else:
        return f"\x1b[36m{s}\x1b[0m"

def install_python_bin(py_path: str, bin_dir: str, name: str, head: str):
    dst_path = os.path.join(bin_dir, name)
    content = (
f"""#!/bin/sh
PYTHONPATH={shlex.quote(py_path)} PYTHONWARNINGS={shlex.quote("ignore")} {head} "$@"
"""
    )
    with open(dst_path, "w") as f:
        print(content, end="", file=f, flush=True)
    os.chmod(dst_path, 0o755)
    print(f"Installed: {cyan(dst_path)}", flush=True)

def main(args):
    home_dir = os.environ["HOME"]
    # bin_dir = os.path.join(home_dir, ".pythia", "bin")
    if args.prefix is not None:
        prefix = args.prefix
    else:
        # prefix = os.path.join(home_dir, ".local")
        prefix = os.path.join(home_dir, ".pythia")
    bin_dir = os.path.join(prefix, "bin")
    lib_dir = os.path.join(home_dir, ".pythia", "lib")
    os.makedirs(bin_dir, exist_ok=True)
    os.makedirs(lib_dir, exist_ok=True)
    py_path = os.path.join(lib_dir, "pythia-dev")
    cwd = os.getcwd()
    os.unlink(py_path)
    os.symlink(cwd, py_path)
    print(f"Current working dir = {cyan(cwd)}")
    print(f"Python path (dev)   = {cyan(py_path)}")
    print(f"Executable prefix   = {cyan(prefix)}")
    install_python_bin(py_path, bin_dir, "autopythia", "python3 -m pythia.auto")
    print("Done installation.")

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--prefix", type=str, default=None)
    args = args.parse_args()
    main(args)
