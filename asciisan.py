#!/usr/bin/env python3

import sys

def main():
    warn = False
    f = open(sys.argv[1], "r", encoding="utf-8")
    for line_idx, line in enumerate(f):
        for char_idx, c in enumerate(line):
            x = ord(c)
            if x >= 128:
                # FIXME: column in units of chars.
                print(f"warning: unicode char = 0x{x:02x} line = {line_idx+1} col = {char_idx+1}")
                warn = False
            elif x < 32 and x not in (0xa, 0xd):
                print(f"warning: control char = 0x{x:02x} line = {line_idx+1} col = {char_idx+1}")
                warn = False
    if not warn:
        print(f"ok")

if __name__ == "__main__":
    main()
