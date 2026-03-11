#!/usr/bin/env python3
"""Read in a driver results JSON file and condense it by deleting certain keys from each run.
"""

import json
import argparse


def condense_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for run in data:
        del run["run_stdout"]
        del run["run_stderr"]
        del run["build_stdout"]
        del run["build_stderr"]
        del run["validation_output"]
        if "sanitize_stdouts" in run:
            del run["sanitize_stdouts"]
        if "sanitize_stderrs" in run:
            del run["sanitize_stderrs"]

    with open(json_file.removesuffix(".json") + "_condensed.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Condense a JSON file by deleting certain keys from each run"
    )
    parser.add_argument("json_file", type=str, help="The JSON file to condense")
    args = parser.parse_args()
    condense_json(args.json_file)
