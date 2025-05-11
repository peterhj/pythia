#!/usr/bin/env python3

import os

def main():
    test_paths = set()
    with open("data/test/interp.txt", "r") as list_file:
        for line in list_file:
            test_paths.add(line.strip())

    list_file = open("data/test/interp.txt", "a")
    new_test_paths = []

    for dirpath, _dirnames, filenames in os.walk("data/test"):
        for filename in filenames:
            if filename.endswith(".py"):
                filepath = os.path.join(dirpath, filename)
                assert filepath.startswith("data/test/")
                filepath = filepath[10:]
                if filepath not in test_paths:
                    print(f"DEBUG: found new path = {filepath}")
                    new_test_paths.append(filepath)
                    test_paths.add(filepath)

    new_test_paths = sorted(new_test_paths)

    for filepath in new_test_paths:
        print(filepath, file=list_file, flush=True)

    list_file.close()

if __name__ == "__main__":
    main()
