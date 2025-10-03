"""
Utility functions for message routing.
"""


from typing import Optional


def get_first_word(input_str: str) -> Optional[str]:
    """
    Finds the first word in a given string, ignoring leading whitespace.

    Returns the first word, or None if there is no first word.
    """
    start = 0

    # Skip leading whitespace
    while start < len(input_str) and input_str[start].isspace():
        start += 1

    # Find end of first word
    end = start
    while end < len(input_str) and not input_str[end].isspace():
        end += 1

    first_word = input_str[start:end]
    return first_word if first_word else None


def is_persona(username: str) -> bool:
    """Returns true if username belongs to a persona"""
    return username.startswith("jupyter-ai-personas")