import os
import json
import pandas as pd

folders = [
    "./data/test/GuardReasoner-VL-7B/",
    "./data/test/GuardReasoner-VL-3B/",
    "./data/test/R-SFT-3B-VLSU/",
    "./data/test/R-SFT-7B-VLSU/",
]
folders = [f for f in folders if os.path.exists(f)]

print(f"\n{'Model':<35} {'Accuracy':>10}  {'Correct/Total':>15}")
print("-" * 65)

for folder in folders:
    model_name = folder.rstrip("/").split("/")[-1]
    file_name = folder + "SIUO/generated_predictions.jsonl"
    if not os.path.exists(file_name):
        print(f"{model_name:<35} {'N/A':>10}")
        continue

    pred = pd.read_json(file_name, lines=True)
    correct = int((pred["predict"] == pred["label"]).sum())
    total = len(pred)
    accuracy = correct / total * 100
    print(f"{model_name:<35} {accuracy:>9.2f}%  {correct}/{total}")

    # per-category breakdown
    categories = pred["category"].unique()
    for cat in sorted(categories):
        cat_df = pred[pred["category"] == cat]
        c = int((cat_df["predict"] == cat_df["label"]).sum())
        t = len(cat_df)
        print(f"  {cat:<33} {c/t*100:>9.2f}%  {c}/{t}")

print()
