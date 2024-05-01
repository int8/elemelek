import hashlib
from collections import Counter
from typing import List, Dict

import language_tool_python

from elemelek.model import InstructionFeature
from elemelek.settings import LANGUAGE_TOOL_CHECK


def calculate_file_md5(file_path: str, block_size=8192):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            hasher.update(block)
    return hasher.hexdigest()


def calculate_text_chunk_md5(texts: List[str]):
    hasher = hashlib.md5()
    for text in texts:
        hasher.update(text.encode())
    return hasher.hexdigest()


def divide_dict(d: dict, n: int) -> List[List[int]]:
    n = max(min(n, len(d)), 1)
    chunks = [list() for _ in range(n)]
    keys = list(d.keys())

    for i, key in enumerate(keys):
        chunks[i % n].append(key)
    return chunks


def count_jsonl_objects(file_path):
    with open(file_path, "r") as file:
        count = sum(1 for _ in file)
    return count


def language_tool_scan(
    instructions: dict, keys: List[int]
) -> Dict[int, InstructionFeature]:
    language_tool = None
    try:
        results = dict()
        language_tool = language_tool_python.LanguageTool(
            "pl-PL",
        )
        for key in keys:
            instruction = instructions[key]
            matches = language_tool.check(instruction.text)
            results[key] = InstructionFeature(
                instruction_id=instruction.id,
                name=LANGUAGE_TOOL_CHECK,
                value=Counter([m.ruleId for m in matches]),
            )

    finally:
        if language_tool:
            language_tool.close()

    return results
