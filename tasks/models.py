import abc
import llama_cpp


class GeneratingAgent(abc.ABC):
    """
    Abstract class encapsulating any agent that can generate text,
    (be it a human, an LLM, a retrieval system ...) to be used in the 
    simulated convesational framework.
    """

    @abc.abstractmethod
    def prompt(self, prompt: str, history: str) -> str:
        """
        Prompt the LLM and get its response.

        :param prompt: The prompt to the LLM.
        :type prompt: str
        :param history: The previous conversational context.
        :type history: str
        :return: The LLM's response.
        :rtype: str
        """
        return ""


class LlamaModel(GeneratingAgent):

    @staticmethod
    def _get_response_from_output(json_output) -> str:
        """
        Extracts the model's response from the raw output as a string.
        Assumes that output is of the form "Q: <prompt> A: <answer>"
        """
        prompt_and_answer = json_output["choices"][0]["text"]
        _, _, answer = prompt_and_answer.partition("A:")
        return answer
    
    @staticmethod
    def get_actual_prompt(prompt: str, history: str):
        return prompt + "\n" + history

    def __init__(self, model: llama_cpp.Llama, max_out_tokens: int, seed: int):
        self.model = model
        self.max_out_tokens = max_out_tokens
        self.seed = seed

    def prompt(self, prompt: str, history: str) -> str:
        full_prompt = self.get_actual_prompt(prompt, history)

        output = self.model.create_completion(
                        prompt=f"Q: {full_prompt} A:",
                        max_tokens=self.max_out_tokens,
                        stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
                        echo=True,
                        seed=self.seed)
        
        response = self._get_response_from_output(output)

        return response
 

