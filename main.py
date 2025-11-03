"""
CLI entry for running the end-to-end PaperQA pipeline.

Usage:
  py -m paperqa.main --q "What is the role of CD33 in AML?"
"""

import argparse
import json
from paperqa.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", "--question", dest="question", type=str, required=True)
    parser.add_argument("--pretty", action="store_true", default=True)
    parser.add_argument("--show-evidence", action="store_true", default=True,
                        help="print top evidence summaries and scores")
    parser.add_argument("--evidence-topk", type=int, default=5,
                        help="number of evidence items to display")
    args = parser.parse_args()

    pipe = Pipeline()
    out = pipe.run(args.question)

    if args.pretty:
        print("\n=== PaperQA Result ===")
        print(f"Question: {out['question']}")
        print("\n-- Background (<=50 words) --")
        print(out["background"].get("background_text", ""))
        # Evidence (optional)
        if args.show_evidence:
            ev_container = out.get("evidence") or {}
            ev_list = ev_container.get("evidence") if isinstance(ev_container, dict) else None
            if ev_list:
                print("\n-- Evidence (top) --")
                for i, ev in enumerate(ev_list[: max(1, args.evidence_topk) ], 1):
                    score = ev.get("score")
                    print(f"[{i}] score={score if score is not None else 'NA'}")
                    print(ev.get("summary", "").strip())
            else:
                print("\n-- Evidence --\n(no evidence available)")
        print("\n-- Answer --")
        ans = out["answer"]
        if isinstance(ans, dict):
            print(ans.get("answer_text") or ans.get("answers") or json.dumps(ans, ensure_ascii=False, indent=2))
        else:
            print(ans)
        print(f"\nLatency: {out['latency_s']} s")
    else:
        print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

