import argparse
import pandas as pd
import time
from pathlib import Path
from aiqs.config import config_from_args
from aiqs.vlm.adjudicate import adjudicate
from aiqs.vlm.state import VLMState, load_vlm_states
from aiqs.eval.decision import CostMatrix, decide_one

PAPER_COST_MATRIX = CostMatrix(10, 3, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--batch_delay", type=float, default=10.0)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = config_from_args(args)
    # Hardfix: results_dir is usually at project root/results
    results_dir = Path("/content/aiqs-agent/results")
    
    def _find_run_dir(res_dir, run_id):
        matches = list(res_dir.glob(f"runs/{run_id}*"))
        if not matches: raise FileNotFoundError(f"No run matching {run_id}")
        return matches[0]

    run_dir = _find_run_dir(results_dir, args.run)
    dec_df = pd.read_csv(run_dir / "decision_scores.csv")
    escalate_items = dec_df[dec_df["final_decision"] == "escalate"]
    
    states = load_vlm_states(run_dir, args.k, escalate_items, results_dir)
    print(f"Processing {len(states)} total calls...")

    results = []
    for i in range(0, len(states), args.batch_size):
        batch = states[i : i + args.batch_size]
        for s in batch:
            v = adjudicate(s)
            p_vlm = v.confidence if v.verdict == "defect" else (1.0 - v.confidence if v.verdict == "clean" else 0.5)
            results.append({
                "image_path": s.image_path, "item_id": s.item_id, "run": s.run_idx, "label": s.label,
                "vlm_verdict": v.verdict, "vlm_conf": v.confidence, "p_vlm": p_vlm,
                "final_decision": decide_one(p_vlm, PAPER_COST_MATRIX).value
            })
        print(f"Finished batch {i//args.batch_size + 1}. Sleeping {args.batch_delay}s...")
        time.sleep(args.batch_delay)

    pd.DataFrame(results).to_csv(run_dir / "vlm_results.csv", index=False)
    print(f"Success! Results saved to {run_dir}/vlm_results.csv")

if __name__ == "__main__":
    main()
