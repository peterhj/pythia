import json

def main():
    log_file = open("data/_journal/_wlog.jsonl", "r")
    line = None
    for line in log_file:
        pass
    if line is None:
        return
    entry = json.loads(line.strip())
    print(entry.keys())
    print(entry["t"])
    print(entry["sort"])
    item = entry["item"]
    print(item.keys())
    print(item["query"])
    if "think" in item and item["think"] is not None:
        print("<think>")
        print(item["think"])
        print("</think>\n")
    print(item["value"])

if __name__ == "__main__":
    main()
