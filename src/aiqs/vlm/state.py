import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class VLMState:
    image_path: str
    item_id: int
    run_idx: int
    label: int
    score: float
    crop_box: Optional[tuple] = None
    run: dict = None

def load_vlm_states(run_dir, k, escalate_items, results_dir):
    states = []
    for idx, row in escalate_items.iterrows():
        # Use 'item_id' column if exists, otherwise fallback to the dataframe index
        item_id = row['item_id'] if 'item_id' in row else idx
        for run_idx in range(k):
            states.append(VLMState(
                image_path=row['image_path'],
                item_id=item_id,
                run_idx=run_idx,
                label=row['label'],
                score=row['score'],
                run={'mock_vlm': False}
            ))
    return states
