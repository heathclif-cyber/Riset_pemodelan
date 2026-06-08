import json
from pathlib import Path

# Load the pruned feature info
with open('experiments/cascade_v2.5_hybrid_pruned_features.json') as f:
    prune_info = json.load(f)

kept = prune_info['kept_features']

# Save as feature_cols_v2.json inside the pruned run folder
run_dir = Path('models/runs/cascade_v2.5_hybrid_pruned')
run_dir.mkdir(parents=True, exist_ok=True)

with open(run_dir / 'feature_cols_v2.json', 'w') as f:
    json.dump(kept, f, indent=2)

print(f'Saved pruned feature_cols_v2.json with {len(kept)} features to:')
print(f'  {run_dir / "feature_cols_v2.json"}')

# Also save a reference copy in the main hybrid folder
main_run = Path('models/runs/cascade_v2.5_hybrid')
with open(main_run / 'feature_cols_pruned_87.json', 'w') as f:
    json.dump(kept, f, indent=2)

print(f'Also saved reference copy to: {main_run / "feature_cols_pruned_87.json"}')
