import pandas as pd
import numpy as np
import os

print("Generating HPO-B style dataset...")
np.random.seed(42)

n_runs = 100000
search_spaces = ['4796', '5527', '5636', '5859', '5860', '5891', '5906', '5965', '5970', '5971']
optimizers = ['adam', 'sgd', 'rmsprop', 'adagrad']
datasets = [f'dataset_{i}' for i in range(50)]

data = {
    'run_id': [f'run_{i:07d}' for i in range(n_runs)],
    'search_space_id': np.random.choice(search_spaces, n_runs),
    'dataset_id': np.random.choice(datasets, n_runs),
    'learning_rate': np.exp(np.random.uniform(np.log(1e-5), np.log(1e-1), n_runs)),
    'batch_size': np.random.choice([16, 32, 64, 128, 256], n_runs),
    'dropout': np.random.uniform(0.0, 0.5, n_runs),
    'num_layers': np.random.randint(1, 6, n_runs),
    'hidden_units': np.random.choice([64, 128, 256, 512], n_runs),
    'optimizer': np.random.choice(optimizers, n_runs),
    'weight_decay': np.exp(np.random.uniform(np.log(1e-6), np.log(1e-2), n_runs)),
    'epochs': np.random.randint(10, 200, n_runs),
    'val_accuracy': np.random.uniform(0.5, 0.99, n_runs),
    'train_loss': np.random.uniform(0.01, 2.0, n_runs),
    'val_loss': np.random.uniform(0.01, 2.5, n_runs),
    'train_time_sec': np.random.uniform(10, 3600, n_runs),
}

df = pd.DataFrame(data)
out_path = '/home/hadoop/hpo_project/data/hpo_runs.csv'
df.to_csv(out_path, index=False)

size_mb = os.path.getsize(out_path) / 1024 / 1024
print(f"✅ Saved {n_runs:,} runs → {out_path}")
print(f"   File size: {size_mb:.1f} MB")
print(f"   Columns: {list(df.columns)}")
print(df.head(3))
