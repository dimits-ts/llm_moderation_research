import collections
import dataclasses
import datetime
import json
import textwrap
import uuid
from typing import Any

import tasks.actors
import tasks.util


class Conversation:
    """
    A class conducting a conversation between different actors (:class:`tasks.actors.Actor`).
    Only one object should be used for a given conversation.
    """

    def __init__(
        self,
        users: list[tasks.actors.IActor],
        moderator: tasks.actors.IActor | None = None,
        history_context_len: int = 5,
        conv_len: int = 5,
    ) -> None:
        """
        Construct the framework for a conversation to take place.

        :param users: A list of discussion participants
        :type users: list[tasks.actors.Actor]
        :param moderator: An actor tasked with moderation if not None, can speak at any point in the conversation , defaults to None
        :type moderator: tasks.actors.Actor | None, optional
        :param history_context_len: How many prior messages are included to the LLMs prompt as context, defaults to 5
        :type history_context_len: int, optional
        :param conv_len: The total length of the conversation (how many times each actor will be prompted), defaults to 5
        :type conv_len: int, optional
        """
        self.users = users
        self.moderator = moderator
        self.conv_len = conv_len
        # unique id for each conversation, generated for persistence purposes
        self.id = uuid.uuid4()

        self.conv_logs = []
        # keep a limited context of the conversation to feed to the models
        self.ctx_history = collections.deque(maxlen=history_context_len)

    def begin_conversation(self, verbose: bool = True) -> None:
        """
        Begin the conversation between the actors.

        :param verbose: whether to print the messages on the screen as they are generated, defaults to True
        :type verbose: bool, optional
        :raises RuntimeError: if the object has already been used to generate a conversation
        """
        if len(self.conv_logs) != 0:
            raise RuntimeError(
                "This conversation has already been concluded, create a new Conversation object."
            )

        for _ in range(self.conv_len):
            for user in self.users:
                self._actor_turn(user, verbose)
                if self.moderator is not None:
                    self._actor_turn(self.moderator, verbose)

    def _actor_turn(self, actor: tasks.actors.IActor, verbose: bool) -> None:
        """
        Prompt the actor to speak and record his response accordingly.

        :param actor: the actor to speak, can be both a user and a moderator
        :type actor: tasks.actors.Actor
        :param verbose: whether to also print the message on the screen
        :type verbose: bool
        """
        res = actor.speak(list(self.ctx_history))

        if len(res.strip()) != 0:
            # append name of actor to his response
            # "user x posted" important for the model to not confuse it with the prompt
            wrapped_res = textwrap.fill(res, 70)
            formatted_res = f"User {actor.get_name()} posted:\n{wrapped_res}"
        else:
            formatted_res = f"<{actor.get_name()} said nothing>"

        if verbose:
            print(formatted_res)

        self.ctx_history.append(formatted_res)
        self.conv_logs.append((actor.get_name(), res))

    def to_dict(self, timestamp_format: str = "%y-%m-%d-%H-%M") -> dict[str, Any]:
        """
        Get a dictionary view of the data and metadata contained in the conversation.

        :param timestamp_format: the format for the conversation's creation time, defaults to "%y-%m-%d-%H-%M"
        :type timestamp_format: str, optional
        :return: a dict representing the conversation
        :rtype: dict[str, Any]
        """
        return {
            "id": str(self.id),
            "timestamp": datetime.datetime.now().strftime(timestamp_format),
            "users": [user.get_name() for user in self.users],
            "user_types": [type(user).__name__ for user in self.users],
            "moderator": (
                self.moderator.get_name() if self.moderator is not None else "None"
            ),
            "moderator_type": (
                type(self.moderator).__name__ if self.moderator is not None else None
            ),
            "user_prompts": [user.describe() for user in self.users],
            "moderator_prompt": (
                self.moderator.describe() if self.moderator is not None else None
            ),
            "ctx_length": len(self.ctx_history),
            "logs": self.conv_logs,
        }

    def to_json_file(self, output_path: str):
        """
        Export the data and metadata of the conversation as a json file.
        Convinience function equivalent to json.dump(self.to_dict(), output_path)

        :param output_path: the path for the exported file
        :type output_path: str
        """
        tasks.util.ensure_parent_directories_exist(output_path)

        with open(output_path, "w", encoding="utf8") as fout:
            json.dump(self.to_dict(), fout, indent=4)

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)


@dataclasses.dataclass
class LLMConvData:
    context: str
    actor_names: list[str]
    actor_attributes: list[list[str]]
    actor_instructions: str
    conv_len: int = 4
    history_ctx_len: int = 4
    moderator_name: str | None = None
    moderator_attributes: list[str] | None = None
    moderator_instructions: str | None = None

    def __post_init__(self):
        assert len(self.actor_names) == len(
            self.actor_attributes
        ), "Number of actor names and actor attribute lists must be the same"

    @staticmethod
    def from_json_file(input_file_path: str):
        with open(input_file_path, "r", encoding="utf8") as fin:
            data_dict = json.load(fin)

        # code from https://stackoverflow.com/questions/68417319/initialize-python-dataclass-from-dictionary
        field_set = {f.name for f in dataclasses.fields(LLMConvData) if f.init}
        filtered_arg_dict = {k: v for k, v in data_dict.items() if k in field_set}
        return LLMConvData(**filtered_arg_dict)

    def to_json_file(self, output_path):
        with open(output_path, "w", encoding="utf8") as fout:
            json.dump(dataclasses.asdict(self), fout, indent=4)
