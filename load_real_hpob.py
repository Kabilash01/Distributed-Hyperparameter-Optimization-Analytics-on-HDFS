import json
import pandas as pd
import os

print("📂 Loading real HPO-B meta-train dataset...")

with open('/home/hadoop/hpo_project/data/hpob-data/meta-train-dataset.json', 'r') as f:
    train_data = json.load(f)

print(f"✅ Loaded! Search spaces: {len(train_data)}")

# Convert nested JSON to flat DataFrame
rows = []
for space_id, datasets in train_data.items():
    for dataset_id, evaluations in datasets.items():
        X = evaluations['X']  # hyperparameter configs
        y = evaluations['y']  # accuracy values
        for i, (config, acc) in enumerate(zip(X, y)):
            row = {
                'run_id': f"{space_id}_{dataset_id}_{i}",
                'search_space_id': space_id,
                'dataset_id': dataset_id,
                'val_accuracy': acc[0] if isinstance(acc, list) else acc
            }
            # Add hyperparameter columns
            for j, val in enumerate(config):
                row[f'hp_{j}'] = val
            rows.append(row)

df = pd.DataFrame(rows)
print(f"✅ Converted to DataFrame: {len(df):,} rows")
print(f"   Search spaces: {df['search_space_id'].nunique()}")
print(f"   Datasets: {df['dataset_id'].nunique()}")
print(f"   Columns: {len(df.columns)}")
print(df.head(3))

# Save as CSV
out_path = '/home/hadoop/hpo_project/data/hpob_real_train.csv'
df.to_csv(out_path, index=False)
size_mb = os.path.getsize(out_path) / 1024 / 1024
print(f"\n✅ Saved to {out_path} ({size_mb:.1f} MB)")
