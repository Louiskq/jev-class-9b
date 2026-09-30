"""Print a few generated text records for review:  python show_samples.py FILE [N]"""
import collections
import json
import sys

path, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2
rs = [json.loads(l) for l in open(path)]
print("=" * 30, path, len(rs), "records")
for r in rs[:n]:
    s = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], indent=1)
    print("\n--- %s | %s | %s | %s" % (r["family"], r["sub"], r["domain"], r.get("spec")))
    if r.get("audit", {}).get("original_state"):
        print("ORIGINAL:", r["audit"]["original_state"][:500])
    print("DOCUMENT:", s[:1600] + (" [...]" if len(s) > 1600 else ""))
    for q in r["questions"]:
        print("  Q (%s): %s" % (q["type"], q["instructions"]))
        if isinstance(q.get("criteria"), dict):
            print("     options:", json.dumps(q["criteria"])[:600])
        elif q.get("criteria"):
            print("     levels:", q["criteria"])
        print("     ANSWER:", q["answer"], "|", (q.get("why") or "")[:200])
print("\nfamilies:", dict(collections.Counter(r["family"] for r in rs)))
print("words:", sorted(len((r["state"] if isinstance(r["state"], str) else json.dumps(r["state"])).split()) for r in rs))
