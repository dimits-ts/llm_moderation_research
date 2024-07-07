import collections
import textwrap
import time

import tasks.actors


class Conversation:

    def __init__(
        self,
        users: list[tasks.actors.Actor],
        moderator: tasks.actors.Actor | None = None,
        history_context_len: int = 5,
        conv_len: int = 5,
    ) -> None:
        self.users = users
        self.moderator = moderator
        self.conv_len = conv_len
        # unique id for each conversation, generated for persistence purposes
        self.id = str(time.time())

        self.conv_logs = []
        # keep a limited context of the conversation to feed to the models
        self.ctx_history = collections.deque(maxlen=history_context_len)

    def begin_conversation(self, verbose: bool = True) -> None:
        if len(self.conv_logs) != 0:
            raise RuntimeError("This conversation has already been concluded, create a new Conversation object.")

        for _ in range(self.conv_len):
            for user in self.users:
                self.actor_turn(user, verbose)
                if self.moderator is not None:
                    self.actor_turn(self.moderator, verbose)

    def actor_turn(self, actor: tasks.actors.Actor, verbose: bool) -> None:
        res = actor.speak(list(self.ctx_history))
        self.conv_logs.append((self.id, actor.get_name(), res))

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
