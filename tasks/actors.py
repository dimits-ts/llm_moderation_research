import tasks.models
import abc


class Actor(abc.ABC):

    @abc.abstractmethod
    def get_name(self) -> str:
        return ""

    @abc.abstractmethod
    def speak(self, history: str) -> str:
        return ""


class LLMActor(Actor):

    def __init__(self, 
                 model: tasks.models.LlamaModel, 
                 name: str,
                 role: str, 
                 attributes: list[str], 
                 context: str,
                 instructions: str) -> None:
        self.model = model
        self.name = name
        self.role = role
        self.attributes = attributes
        self.context = context
        self.instructions = instructions

    def _actor_prompt(self, history: str) -> str:
        prompt = f"""
        Your name is {self.name}, and are a {self.role}.
        The following statements describe you and your behavior: {", ".join(self.attributes)}
        The scenario you are participating in: {self.context} 
        What you need to do: {self.instructions}
        The conversation so far:\n {history}
        """

        return prompt
    
    def speak(self, history: str) -> str:
        actor_prompt = self._actor_prompt(history)
        response = self.model.prompt(actor_prompt, history)
        return response
    
    def get_name(self) -> str:
        return self.name