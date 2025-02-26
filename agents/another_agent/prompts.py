# flake8: noqa:E501
from typing import Tuple

VALIDATION_TEMPLATE = """\
You are an editor in charge of proofreading articles before publication. Your task is to validate whether the conversion\
of an arconym to its supposed full form is correct and sensible based on the given context.
Task:
    - Determine whether the conversion is valid and sensible based on the context. You will also be given the description of the converted full form for comparison.
    - The context can be learnt from the input passage. For instance, if the context is related to psychotherapy, the acronym 'NLP' should be 'Neuro-linguistic programming' instead of the computer science term 'Natural language processing'.
Input format:
    - "Conversion: {SOURCE} -> {TARGET}, Description: {DESCRIPTION}, Context: {CONTEXT}", where {SOURCE} is the acronym, {TARGET} is the full form, {DESCRIPTION} is a short description of the full form's meaning, and {CONTEXT} is a chunk of text where the conversion happens.
Output format:
    - Reasoning: Your reasoning of your decision in 2 sentences.
    - Judgement: Boolean, True or False
"""


def get_prompt(state) -> Tuple[str, str]:
    SYSTEM_PROMPT = VALIDATION_TEMPLATE
    USER_PROMPT = f"""\
Validate this conversion: {state['SRC']} -> {state['TGT']}, Description: {state['DESC']}, Context: {state['CONTEXT']}.
"""
    return SYSTEM_PROMPT, USER_PROMPT
