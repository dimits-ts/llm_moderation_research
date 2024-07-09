from unittest import TestCase
from tasks.conversation import LLMConvData


class TestLLMConvData(TestCase):
    def test_from_json_file(self):
        data = LLMConvData.from_json_file("output/test1.json")
        assert len(data.actor_names) != 0
        assert data.moderator_name is not None
        print(data)

    def test_to_json_file(self):
        data = LLMConvData(context="You are in an online chatroom. You see the following post on a social media site:"
                                   "'Homosexuality, transgender sex all are deviations from normal human behavior "
                                   "as well.'",
                           actor_names=["Steve2001", "GeorgeBush78"],
                           actor_attributes=[
                               ["African American"],
                               ["Typical", "average", "white", "American"]
                           ],
                           actor_instructions="Write a comment responding to the post. Insist on your opinion"
                                              " unless convinced by other users."
                                              " Try to follow the instructions of the moderator, if he intervenes."
                                              " You are allowed to act frustrated or aggresively if provoked.",
                           moderator_name="moderator01",
                           moderator_attributes=["just", "strict"],
                           moderator_instructions="Intervene if one user dominates or veers off-topic. "
                                                  "Respond only if necessary. "
                                                  "Write '<No response>' if intervention is unecessary."
                                                  " Be firm and threaten to displine non-cooperating users.")
        data.to_json_file("output/test1.json")
