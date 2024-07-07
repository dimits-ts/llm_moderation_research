import abc
import typing

import tasks.models


class Actor(abc.ABC):

    @abc.abstractmethod
    def get_name(self) -> str:
        return ""

    @abc.abstractmethod
    def speak(self, history: list[str]) -> str:
        return ""

    @abc.abstractmethod
    def describe(self):
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
        prompt = f"You are {self.name} a {",".join(self.attributes)} user. {self.context} {self.instructions}."
        return {"role": "system", "content": prompt}

    def _message_prompt(self, history: list[str]) -> dict:
        return {
            "role": "user",
            "content": "\n".join(history) + f"\n{self.get_name()}:"
        }

    def describe(self):
        return f"Model: {type(self.model).__name__}. Prompt: {self._system_prompt()["content"]}"


    @typing.final
    def speak(self, history: list[str]) -> str:
        system_prompt = self._system_prompt()
        message_prompt = self._message_prompt(history)
        response = self.model.prompt([system_prompt, message_prompt])
        return response

    @typing.final
    def get_name(self) -> str:
        return self.name
