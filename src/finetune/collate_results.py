#!/usr/bin/env python3
import os
import json
import re
import csv
import argparse
from collections import defaultdict


def parse_training_comparison(file_path, task_name):
    """Extract compression rate and variance from training_comparison.txt"""
    metrics = {}
    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Extract language code from the file path
        path_parts = file_path.split("/")
        lang_idx = path_parts.index(task_name) + 1 if task_name in path_parts else -1
        if lang_idx >= 0 and lang_idx < len(path_parts):
            lang = path_parts[lang_idx]
        else:
            # Try to extract from contents if path doesn't work
            match = re.search(r"Post-training test_(\w+)_compression_rate", content)
            lang = match.group(1) if match else "unknown"

        # Extract compression rate
        rate_match = re.search(
            f"Post-training test_{lang}_compression_rate: ([0-9.]+)", content
        )
        if rate_match:
            metrics[f"com_rate_{lang}"] = float(rate_match.group(1))

        # Extract compression variance
        var_match = re.search(
            f"Post-training test_{lang}_compression_var: ([0-9.]+)", content
        )
        if var_match:
            metrics[f"com_var_{lang}"] = float(var_match.group(1))

        return metrics
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {}


def find_all_languages(task_dir):
    """Find all languages in the task directory by looking at folder structure"""
    languages = set()

    try:
        for entry in os.listdir(task_dir):
            lang_dir = os.path.join(task_dir, entry)
            if os.path.isdir(lang_dir):
                languages.add(entry)
    except Exception as e:
        print(f"Error finding languages in {task_dir}: {e}")

    return languages


def extract_results(task_dir, all_languages, task_name):
    """Walk through directories and extract metrics for completed experiments"""
    all_results = []

    for root, dirs, files in os.walk(task_dir):
        # Check if this directory has both required files
        if "results.json" in files and "training_comparison.txt" in files:
            result_row = {}

            path_parts = root.split("/")
            if task_name in path_parts:
                task_idx = path_parts.index(task_name)
                if task_idx + 1 < len(path_parts):
                    lang = path_parts[task_idx + 1]

                    try:
                        with open(os.path.join(root, "results.json"), "r") as f:
                            results_data = json.load(f)
                            if (
                                "test_accuracy" in results_data
                                and "accuracy" in results_data["test_accuracy"]
                            ):
                                result_row[f"acc_{lang}"] = results_data[
                                    "test_accuracy"
                                ]["accuracy"]
                    except Exception as e:
                        print(f"Error reading results.json in {root}: {e}")

                    training_metrics = parse_training_comparison(
                        os.path.join(root, "training_comparison.txt"), task_name
                    )
                    result_row.update(training_metrics)

                    result_row["experiment_path"] = root
                    result_row["language"] = lang  # Add language info for debugging

                    all_results.append(result_row)
    return all_results


def merge_results(all_results):
    """Merge results from different experiments into a comprehensive dataset"""
    merged = defaultdict(dict)

    for result in all_results:
        exp_path = result.get("experiment_path", "")

        # Find the experiment base path (everything before the task name)
        path_parts = exp_path.split("/")

        # Find where the task directory starts and use everything before it as experiment ID
        task_start_idx = -1
        for i, part in enumerate(path_parts):
            if part in ["sib200", "xnli", "pawsx"]:  # Add your task names here
                task_start_idx = i
                break

        if task_start_idx > 0:
            # Use the path up to the task directory as the experiment identifier
            exp_id = "/".join(path_parts[:task_start_idx])
        else:
            # Fallback to timestamp matching
            exp_id = None
            for part in path_parts:
                if re.match(r".*\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}.*", part):
                    exp_id = part
                    break
            if not exp_id:
                continue

        # Merge all metrics for this experiment into a single row
        for key, value in result.items():
            if key not in ["experiment_path", "language"]:
                merged[exp_id][key] = value

    return list(merged.values())


def generate_csv(results, languages, output_path):
    """Generate CSV file with the collected metrics ordered by a fixed language order"""
    if not results:
        print("No results to write")
        return

    # Fixed language order
    fixed_order = ["en", "es", "ru", "uk", "hi", "te", "ur"]

    # Filter the fixed order to include only the languages present in the task
    ordered_languages = [lang for lang in fixed_order if lang in languages]

    # Create ordered column names: all accuracies, then compression rates, then variances
    fieldnames = []
    for lang in ordered_languages:
        fieldnames.append(f"acc_{lang}")
    for lang in ordered_languages:
        fieldnames.append(f"com_rate_{lang}")
    for lang in ordered_languages:
        fieldnames.append(f"com_var_{lang}")

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Fill in missing values with "-"
            filled_row = {}
            for field in fieldnames:
                filled_row[field] = row.get(field, "-")
            writer.writerow(filled_row)
    print(f"Results written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collate experimental results into a CSV file"
    )
    parser.add_argument("--task_dir", help="Directory for the task (e.g., xnli)")
    args = parser.parse_args()

    task_name = os.path.basename(args.task_dir)
    output_path = os.path.join(args.task_dir, f"{task_name}_results.csv")

    print(f"Processing experiments in {args.task_dir}")

    all_languages = find_all_languages(args.task_dir)
    print(f"Found language folders: {', '.join(sorted(all_languages))}")

    # Extract results from completed experiments
    task_name = args.task_dir.split("/")[-1]
    all_results = extract_results(args.task_dir, all_languages, task_name)
    print(f"Found {len(all_results)} completed experiments")

    merged_results = merge_results(all_results)
    generate_csv(merged_results, all_languages, output_path)


if __name__ == "__main__":
    main()
