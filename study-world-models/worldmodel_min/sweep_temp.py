"""复现论文 Table: 温度 τ 如何影响「梦里成绩」与「真实成绩」的落差"""
import json, numpy as np
from world_model import stage_controller
rows = []
for tau in [0.25, 0.5, 1.0, 1.15, 1.5]:
    print("="*70); print(f"τ = {tau}")
    real, dream = stage_controller(generations=60, temperature=tau, pop=32, n_seeds=4)
    rows.append((tau, dream, real))
print("\n\n===== 温度扫描汇总 =====")
print(f"{'τ':>6} | {'梦中得分':>10} | {'真实得分':>10} | {'落差(梦-真)':>12}")
for tau, d, r in rows:
    print(f"{tau:>6} | {d:>10.1f} | {r:>10.1f} | {d-r:>12.1f}")
json.dump(rows, open("sweep.json","w"), indent=2)
