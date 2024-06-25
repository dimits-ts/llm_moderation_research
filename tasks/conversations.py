def get_response_from_output(json_output) -> str:
    """
    Extracts the model's response from the raw output as a string.
    Assumes that output is of the form "Q: <prompt> A: <answer>"
    """
    prompt_and_answer = json_output["choices"][0]["text"]
    _, _, answer = prompt_and_answer.partition("A:")
    return answer