import tasks.actors
import pandas as pd

import collections
import time


class ConversationManager:

    def __init__(self, actors: list[tasks.actors.Actor], history_len: int=5, conv_len: int=5) -> None:
        self.actors = actors
        self.history_len = history_len
        self.conv_len = conv_len

    def begin_conversation(self, verbose: bool=True) -> pd.DataFrame:
        # unique id for each conversation, generated for persisence purposes
        conv_id = time.time()

        total_history = []
        # counter to determine the ordering of the responses in the dataframe
        conv_counter = 0
        # keep a limited context of the conversation
        history = collections.deque(maxlen=self.history_len)

        for _ in range(self.conv_len):
            for actor in self.actors:
                conv_counter += 1
                history_str = " ".join(history)
                res = actor.speak(history_str)

                if len(res.strip()) != 0: 
                    total_history.append((conv_id, conv_counter, actor.get_name(), res))
                    
                    # append name of actor to his response
                    # "user x posted" important for the model to not confuse it with the prompt
                    named_res = f"User {actor.get_name()} posted:\n{res}"
                    history.append(named_res)

                    if verbose:
                        print(named_res)
                else:
                    if verbose:
                        print(f"<{actor.get_name()} said nothing>")

        return pd.DataFrame(total_history, 
                            columns=["conv_id", "response_order", "actor_name", "contents"])

