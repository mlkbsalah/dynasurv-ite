import argparse
import glob
import json
import os

SUBTYPE = "HR+HER2-"
N_LINES = 4


def aggregate_results(seed, date):
    # read all trial result files
    result_dir = f"../models/{SUBTYPE}/{N_LINES}lines/seed_{seed}_{date}"
    files = glob.glob(f"{result_dir}/trial_*.json")
    all_results = [json.load(open(f)) for f in files]

    # find the best trial based on mean_ci
    best_trial = max(all_results, key=lambda x: x["mean_ci"])
    with open(f"{result_dir}/best_config.json", "w") as f:
        json.dump(best_trial, f, indent=2)

    # save all trials in a single file for reference
    with open(f"{result_dir}/all_trials.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # clean up individual trial files
    for f in files:
        os.remove(f)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--date", type=str, required=True)
    args = parser.parse_args()

    # aggregate results
    aggregate_results(args.seed, args.date)
