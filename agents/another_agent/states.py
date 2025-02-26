from typing import Dict, TypedDict


class ValidateState(TypedDict):
    SRC: str
    TGT: str
    DESC: str
    SEN: str
    CONTEXT: str
    external_args: Dict


class OutputState(TypedDict):
    response: str


class InternalState(ValidateState, OutputState):
    pass
