import openml
import json
import pandas as pd

# Download HPO runs from OpenML
print("Downloading from OpenML...")
openml.config.apikey = ""

# Get study 218 which contains HPO-B runs
study = openml.study.get_suite(218)
print(f"Tasks: {len(study.tasks)}")

# Download first 50 tasks
runs_data = []
for task_id in list(study.tasks)[:50]:
    try:
        runs = openml.runs.list_runs(task=[task_id], size=100)
        runs_data.extend(list(runs.values()))
        print(f"Task {task_id}: {len(runs)} runs")
    except:
        pass

df = pd.DataFrame(runs_data)
df.to_csv("/home/hadoop/hpo_project/data/hpob_runs.csv", index=False)
print(f"Saved {len(df)} runs to hpob_runs.csv")
