#!/usr/bin/env python3

import sys

def main():
    warn = 0
    f = open(sys.argv[1], "r", encoding="utf-8")
    for line_idx, line in enumerate(f):
        for char_idx, c in enumerate(line):
            x = ord(c)
            if warn > 10:
                warn += 1
            elif warn == 10:
                print(f"warning: ...")
                warn += 1
            elif x >= 127:
                # FIXME: column in units of chars.
                print(f"warning: non-ascii char = 0x{x:02x} line = {line_idx+1} col = {char_idx+1}")
                warn += 1
            elif x < 32 and x not in (0xa, 0xd):
                print(f"warning: control char = 0x{x:02x} line = {line_idx+1} col = {char_idx+1}")
                warn += 1
    if warn > 0:
        print(f"warning: total {warn} non-ascii or control chars")
    else:
        print(f"ok")

if __name__ == "__main__":
    main()
