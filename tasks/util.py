import os


def ensure_parent_directories_exist(output_path: str) -> None:
    """
    Create all parent directories if they do not exist.
    :param output_path: the path for which parent dirs will be generated
    """
    # Extract the directory path from the given output path
    directory = os.path.dirname(output_path)

    # Create all parent directories if they do not exist
    if directory:
        os.makedirs(directory, exist_ok=True)


def generate_datetime_filename(output_dir: str=None, format: str="%y-%m-%d-%H-%M") -> str:
    datetime_name = datetime.datetime.now().strftime(format)

    if output_dir is None:
        return datetime_name
    else:
        return os.path.join(output_dir, datetime_name)