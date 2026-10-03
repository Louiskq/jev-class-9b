"""Show splits and the first row's fields of candidate public datasets (streaming; light)."""
from datasets import get_dataset_split_names, load_dataset


def show(name, cfg=None):
    try:
        splits = get_dataset_split_names(name, cfg)
        ex = next(iter(load_dataset(name, cfg, split=splits[0], streaming=True)))
        print("== %s %s splits=%s" % (name, cfg or "", splits))
        for k, v in ex.items():
            print("   %s: %s" % (k, repr(v)[:300]))
    except Exception as e:
        print("== %s %s: ERROR %s: %s" % (name, cfg, type(e).__name__, str(e)[:200]))


show("openai/gsm8k", "main")
show("tasksource/proofwriter")
show("lukaemon/bbh", "date_understanding")
show("lukaemon/bbh", "logical_deduction_five_objects")
show("TAUR-Lab/MuSR")
