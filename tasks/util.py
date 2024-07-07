import os
import datetime


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


def generate_datetime_filename(
    output_dir: str = None, format: str = "%y-%m-%d-%H-%M"
) -> str:
    """
    Generate a filename based on the current date and time.

    :param output_dir: The path to the generated file, defaults to None
    :type output_dir: str, optional
    :param format: strftime format, defaults to "%y-%m-%d-%H-%M"
    :type format: str, optional
    :return: the full path for the generated file
    :rtype: str
    """
    datetime_name = datetime.datetime.now().strftime(format)

    if output_dir is None:
        return datetime_name
    else:
        return os.path.join(output_dir, datetime_name)
