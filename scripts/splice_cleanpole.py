#!/usr/bin/env python3
"""Splice clean-pole Thinking explicit_no_cot rows into the canonical task6 table.

Produces a task6 table identical to the canonical n=50 deep table except that the
Thinking `explicit_no_cot` rows are replaced with the clean NoThinking-pole rows.
Everything else (Thinking cot/neutral, all Instruct rows) is byte-for-byte the
published data, so compute_hcds.py on the output yields the 6-feature HCDS under
the clean pole, directly comparable to the published contaminated-pole number.

Match key: (model, prompt_condition, sample_idx). The clean table is expected to
contain only Thinking explicit_no_cot rows; any non-matching rows in it are an
error (we want a surgical swap, not a merge of unrelated cells).
"""
import argparse, csv, sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--clean", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    canon = list(csv.DictReader(args.canonical.open()))
    clean = list(csv.DictReader(args.clean.open()))
    if not canon:
        sys.exit("canonical table empty")
    fields = list(canon[0].keys())

    # Index clean rows by match key; sanity-check they are all Thinking no_cot.
    clean_by_key = {}
    for r in clean:
        if r["model"] != "thinking" or r["prompt_condition"] != "explicit_no_cot":
            sys.exit(f"unexpected clean row: model={r['model']} cond={r['prompt_condition']}")
        clean_by_key[(r["model"], r["prompt_condition"], int(r["sample_idx"]))] = r

    out_rows = []
    swapped = 0
    for r in canon:
        key = (r["model"], r["prompt_condition"], int(r["sample_idx"]))
        if key in clean_by_key:
            cr = clean_by_key[key]
            # Keep canonical schema/order; pull every shared column from the clean row.
            merged = {f: cr.get(f, r.get(f, "")) for f in fields}
            out_rows.append(merged)
            swapped += 1
        else:
            out_rows.append(r)

    expected = sum(
        1 for r in canon if r["model"] == "thinking" and r["prompt_condition"] == "explicit_no_cot"
    )
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(
        f"spliced {swapped}/{expected} thinking explicit_no_cot rows "
        f"(clean rows available: {len(clean_by_key)}); wrote {len(out_rows)} rows to {args.out}"
    )
    if swapped != expected:
        print("WARNING: swapped count != expected canonical no_cot rows", file=sys.stderr)


if __name__ == "__main__":
    main()
