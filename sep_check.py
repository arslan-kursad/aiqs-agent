import pandas as pd
from collections import Counter
from aiqs.eval.decision import CostMatrix, Decision, decide_one

RUN = "results/runs/patchcore-wide_resnet50_2_mvtec-capsule_20260623T142659Z"
df = pd.read_csv(f"{RUN}/mock_vlm_results.csv")
LOCKED, REALISTIC = CostMatrix(10, 3, 1), CostMatrix(100, 3, 1)
K = df["run"].nunique()

# (a) per-item verdict stability
print("=== (a) per-item verdict stability (across K runs) ===")
ft = Counter()
for img, g in df.groupby("image_path"):
    vs = sorted(set(g["vlm_verdict"]))
    ft["stable: " + vs[0] if len(vs) == 1 else "FLIP: " + "/".join(vs)] += 1
for k, v in sorted(ft.items()):
    print(f"  {k}: {v}")

# (b)+(c) escapes = defective (label 1) auto-PASSed: identity + stability + conf band
esc = df[(df["label"] == 1) & (df["final_decision"] == "pass")]
print(f"\n=== (b)(c) escapes (wrong-PASS) - {len(esc)} rows over {K} runs ===")
band = lambda c: ">=0.98" if c >= 0.98 else "0.80-0.98" if c >= 0.80 else "<0.80"
for img, g in esc.groupby("image_path"):
    runs_item = int((df["image_path"] == img).sum())
    confs = [round(c, 2) for c in g["vlm_conf"]]
    print(f"  {img.split('/')[-1]}: escaped {len(g)}/{runs_item} runs   "
          f"conf={confs}  band={[band(c) for c in g['vlm_conf']]}")

# (d) re-decide under 100/3/1 on STORED p_vlm - MECHANISM ONLY (not a cost headline)
print("\n=== (d) re-decide: rescue-loss vs escape-kill (mechanism, NOT cost) ===")
for name, cost in [("10/3/1 (as-run)", LOCKED), ("100/3/1", REALISTIC)]:
    dec = df["p_vlm"].apply(lambda p: decide_one(float(p), cost).value)
    resc = int(((df["label"] == 0) & (dec == "pass")).sum()) / K
    escp = int(((df["label"] == 1) & (dec == "pass")).sum()) / K
    abst = int((dec == "escalate").sum()) / K
    print(f"  {name}: rescued/run={resc:.1f}  escapes/run={escp:.1f}  abstain/run={abst:.1f}")
