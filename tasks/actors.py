import tasks.models
import abc
import typing


class Actor(abc.ABC):

    @abc.abstractmethod
    def get_name(self) -> str:
        return ""

    @abc.abstractmethod
    def speak(self, history: str) -> str:
        return ""


class AbstractLlmActor(Actor, abc.ABC):

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

    @abc.abstractmethod
    def _actor_prompt(self, history: str) -> str:
        return ""
        
    
    @typing.final
    def speak(self, history: str) -> str:
        actor_prompt = self._actor_prompt(history)
        response = self.model.prompt(actor_prompt)
        return response
    
    @typing.final
    def get_name(self) -> str:
        return self.name
    

class SmartLlmActor(AbstractLlmActor):
    def __init__(self, 
                 model: tasks.models.LlamaModel, 
                 name: str,
                 role: str, 
                 attributes: list[str], 
                 context: str,
                 instructions: str) -> None:
        super().__init__(model, name, role, attributes, context, instructions)

    
    def _actor_prompt(self, history: str) -> str:
        prompt = f"""
        Your name is {self.name}, and are a {self.role}.
        The following statements describe you and your behavior: {", ".join(self.attributes)}
        The scenario you are participating in: {self.context} 
        What you need to do: {self.instructions}
        The conversation so far:\n {history}
        """

        return prompt
    

class DumbLlmActor(AbstractLlmActor):
    def __init__(self, 
                 model: tasks.models.LlamaModel, 
                 name: str,
                 role: str, 
                 attributes: list[str], 
                 context: str,
                 instructions: str) -> None:
        super().__init__(model, name, role, attributes, context, instructions)

    # make prompt as small as humanly possible
    def _actor_prompt(self, history: str) -> str:
        prompt = f"{self.context} {self.instructions} \nThe conversation so far: {history}"
        return prompt
    