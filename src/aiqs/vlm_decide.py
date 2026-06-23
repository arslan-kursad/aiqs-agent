import argparse
import pandas as pd
import time
from pathlib import Path
from aiqs.vlm.adjudicate import adjudicate
from aiqs.vlm.state import VLMState
from aiqs.eval.decision import CostMatrix, decide_one

PAPER_COST_MATRIX = CostMatrix(10, 3, 1)

def run_vlm(run_dir, k, batch_size=5, batch_delay=10):
    dec_df = pd.read_csv(run_dir / "decision_scores.csv")
    escalate_items = dec_df[dec_df["final_decision"] == "escalate"]
    results = []

    print(f"Found {len(escalate_items)} items to adjudicate. Total calls: {len(escalate_items)*k}")
    for _, item in escalate_items.iterrows():
        for run_idx in range(k):
            state = VLMState(
                image_path=item['image_path'],
                detector_score=item['score'],
                detector_p=item['p_calibrated'] if 'p_calibrated' in item else item['score'],
                label=item['label']
            )

            verdict = adjudicate(state)
            p_vlm = 1.0 if verdict.verdict == 'defect' else 0.0 if verdict.verdict == 'clean' else 0.5

            results.append({
                'image_path': item['image_path'],
                'run': run_idx,
                'label': item['label'],
                'vlm_verdict': verdict.verdict,
                'vlm_conf': verdict.confidence,
                'p_vlm': p_vlm,
                'final_decision': decide_one(p_vlm, PAPER_COST_MATRIX).value
            })

            if len(results) % batch_size == 0:
                print(f"Completed {len(results)} calls. Throttling {batch_delay}s...")
                time.sleep(batch_delay)

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--batch_delay", type=float, default=10)
    args = parser.parse_args()
    
    # Resolve run directory
    base_path = Path("results/runs")
    run_dir = next(base_path.glob(f"{args.run}*"))
    
    df = run_vlm(run_dir, args.k, args.batch_size, args.batch_delay)
    df.to_csv(run_dir / "vlm_results.csv", index=False)
    print(f"Success! Results saved to {run_dir}/vlm_results.csv")

if __name__ == '__main__':
    main()