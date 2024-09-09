import pandas as pd

import os
import json
import re


def import_conversations(conv_dir: str) -> pd.DataFrame:
    """
    Import conversation data from a directory containing JSON files and convert them to a DataFrame.

    Recursively reads all JSON files from the specified directory,
    and extracts relevant fields. It also adds metadata about the conversation variant.

    :param conv_dir: Path to the directory containing the conversation JSON files.
    :type conv_dir: str
    :return: A DataFrame with conversation data, including the ID, user prompts, messages,
             and conversation variant.
    :rtype: pd.DataFrame

    :example:
        >>> df = import_conversations("/path/to/conversation/data")
    """
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
    """
    Import annotation data from a directory containing JSON files and convert them to a DataFrame.

    Recursively reads all JSON files from the specified directory, and extracts relevant fields. Also parses annotator
    attributes and toxicity values from the logs.

    :param annot_dir: Path to the directory containing the annotation JSON files.
    :type annot_dir: str
    :return: A DataFrame with annotation data, including conversation ID, annotator prompts,
             messages, and toxicity values.
    :rtype: pd.DataFrame

    :example:
        >>> df = import_annotations("/path/to/annotation/data")
    """
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
    """
    Recursively list all files in a directory and its subdirectories.

    :param start_path: The starting directory path. Defaults to the current directory.
    :type start_path: str, optional
    :return: A list of file paths.
    :rtype: list[str]

    :example:
       >>> file_paths = _files_from_dir_recursive("/path/to/data")
   """
    all_files = []
    for root, dirs, files in os.walk(start_path):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


# code adapted from ChatGPT
def _extract_attributes(text: str) -> str | None:
    """
    Extract annotator attributes from a given text using a regular expression.

    This function searches for the pattern "You are (something) expert annotator" in the
    provided text and returns the matched group. If no match is found, it returns None.

    :param text: The input string containing annotator information.
    :type text: str
    :return: The extracted annotator attribute, or None if no match is found.
    :rtype: str | None

    :example:
        >>> attribute = _extract_attributes("You are a linguistics expert annotator")
        >>> print(attribute)  # Output: "a linguistics"
    """
    # Regex pattern to match the desired attributes
    pattern = r"You are (.+?) expert annotator"
    match = re.search(pattern, text)
    if match:
        return f"{match.group(1)}"
    return None


def _extract_toxicity_value(text: str) -> str | None:
    """
   Extract toxicity value from a given text using a regular expression.

   This function searches for the pattern "Toxicity=<number>" in the provided text and
   returns the toxicity value as a string. If no match is found, it returns None.

   :param text: The input string containing toxicity information.
   :type text: str
   :return: The extracted toxicity value, or None if no match is found.
   :rtype: str | None

   :example:
       >>> toxicity = _extract_toxicity_value("Toxicity=4.5")
       >>> print(toxicity)  # Output: "4.5"
   """
    # Regex pattern to match "Toxicity=<number>"
    pattern = r"Toxicity=(\d+\.?\d*)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None