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

def install_python_bin(cwd: str, prefix: str, name: str, head: str):
    dst_path = os.path.join(prefix, "bin", name)
    content = (
f"""#!/bin/sh
PYTHONPATH={shlex.quote(cwd)} PYTHONWARNINGS={shlex.quote("ignore")} {head} "$@"
"""
    )
    with open(dst_path, "w") as f:
        print(content, end="", file=f, flush=True)
    os.chmod(dst_path, 0o755)
    print(f"Installed: {cyan(dst_path)}", flush=True)

def main(args):
    if args.prefix is not None:
        prefix = args.prefix
    else:
        prefix = os.path.join(os.environ["HOME"], ".local")
    cwd = os.getcwd()
    print(f"Current working dir = {cyan(cwd)}")
    print(f"Installation prefix = {cyan(prefix)}")
    qqq_src_path = os.path.join(cwd, "qqq_vim.py")
    install_python_bin(cwd, prefix, "_qqq_vim", f"python3 {shlex.quote(qqq_src_path)}")
    qq_src_path = os.path.join(cwd, "qq_vim.py")
    install_python_bin(cwd, prefix, "_qq_vim", f"python3 {shlex.quote(qq_src_path)}")
    print("Done installation.")

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--prefix", type=str, default=None)
    args = args.parse_args()
    main(args)
