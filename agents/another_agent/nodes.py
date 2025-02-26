from typing import Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.another_agent import states
from agents.another_agent.prompts import get_prompt


class Nodes:

    def __init__(self, model_args: Dict[str, str]) -> None:
        self.model = ChatOpenAI(**model_args)

    def validate_node(self, state: states.ValidateState) -> states.OutputState:
        system_prompt, user_prompt = get_prompt(state)
        response = self.model.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )

        return {"response": response.content, "inputs": state}


# def validate_node(state: states.ValidateState) -> states.OutputState:
#     return {"response": "test", "inputs": state}
