import pandas as pd

import os
import json
import re


def import_conversations(conv_dir: str) -> pd.DataFrame:
    file_paths = _files_from_dir_recursive(conv_dir)
    rows = []

    for file_path in file_paths:
        with open(file_path, "r") as fin:
            conv = json.load(fin)

        conv = pd.json_normalize(conv)
        conv = conv[["id", "user_prompts", "logs"]]
        conv = conv.explode("logs")
        # get name, not path of parent directory
        conv["conv_variant"] = os.path.basename(os.path.dirname(file_path))
        conv["user"] = conv.logs.apply(lambda x: x[0])
        conv["message"] = conv.logs.apply(lambda x: x[1])
        del conv["logs"]
        rows.append(conv)

    full_df = pd.concat(rows)
    full_df = full_df.set_index("id")
    return full_df


def import_annotations(annot_dir: str) -> pd.DataFrame:
    file_paths = _files_from_dir_recursive(annot_dir)
    rows = []

    for file_path in file_paths:
        with open(file_path, "r") as fin:
            conv = json.load(fin)

        conv = pd.json_normalize(conv)
        conv = conv[["conv_id", "annotator_prompt", "logs"]]
        conv = conv.explode("logs")
        conv.annotator_prompt = conv.annotator_prompt.apply(_extract_attributes)
        conv["message"] = conv.logs.apply(lambda x: x[0])
        conv["toxicity"] = conv.logs.apply(lambda x: x[1])
        conv["toxicity"] = conv.toxicity.apply(_extract_toxicity_value)
        del conv["logs"]
        rows.append(conv)

    full_df = pd.concat(rows)
    full_df = full_df.set_index("conv_id")
    return full_df


# code adapted from https://www.geeksforgeeks.org/python-list-all-files-in-directory-and-subdirectories/
def _files_from_dir_recursive(start_path="."):
    all_files = []
    for root, dirs, files in os.walk(start_path):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


# code adapted from ChatGPT
def _extract_attributes(text):
    # Regex pattern to match the desired attributes
    pattern = r"You are (.+?) expert annotator"
    match = re.search(pattern, text)
    if match:
        return f"{match.group(1)}"
    return None


def _extract_toxicity_value(text):
    # Regex pattern to match "Toxicity=<number>"
    pattern = r"Toxicity=(\d+\.?\d*)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None
