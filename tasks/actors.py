import tasks.models
import abc
import typing


class Actor(abc.ABC):

    @abc.abstractmethod
    def get_name(self) -> str:
        return ""

    @abc.abstractmethod
    def speak(self, history: list[str]) -> str:
        return ""


class LlmActor(Actor):

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

    
    def _system_prompt(self) -> dict:
        prompt = f"""
        You are {self.name} a {",".join(self.attributes)} user. {self.context} {self.instructions}.'
        """ 
        return {"role": "system", "content": prompt}
    
    def _message_prompt(self, message: str) -> dict:
        return {"role": "user", "content": message}

    @typing.final
    def speak(self, history: list[str]) -> str:
        system_prompt = self._system_prompt()
        messages = []
        for message in history:
            message_prompt = self._message_prompt(message)
            messages.append((system_prompt, message_prompt))
        
        response = self.model.prompt(messages)

        return response

    @typing.final
    def get_name(self) -> str:
        return self.name

    