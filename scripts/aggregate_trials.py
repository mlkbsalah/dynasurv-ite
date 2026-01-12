import argparse, glob, json

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

result_dir = f"../models/HR+HER2-/4lines/seed_{args.seed}"
files = glob.glob(f"{result_dir}/trial_*.json")

all_results = [json.load(open(f)) for f in files]
best_trial = max(all_results, key=lambda x: x["mean_ci"])

with open(f"{result_dir}/best_config.json", "w") as f:
    json.dump(best_trial, f, indent=2)

print("Best trial:", best_trial["trial_id"], "Mean CI:", best_trial["mean_ci"])