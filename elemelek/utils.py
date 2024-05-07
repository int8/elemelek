import hashlib
import math

from collections import Counter
from typing import List, Dict, Sequence

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


def get_samples_per_group(groups: List[Sequence], k: int) -> List[int]:
    num_groups = len(groups)
    ideal_samples_per_group = math.floor(k / num_groups)

    num_per_group = [min(len(g), ideal_samples_per_group) for g in groups]

    total_samples_from_uniform = sum(num_per_group)

    remaining_samples = k - total_samples_from_uniform

    while remaining_samples > 0:
        for i, group in enumerate(groups):
            if num_per_group[i] < len(group) and remaining_samples > 0:
                num_per_group[i] += 1
                remaining_samples -= 1
            if remaining_samples <= 0:
                break

    return num_per_group


def compute_df_row_md5(row):
    str_to_hash = "".join(row)
    result = hashlib.md5(str_to_hash.encode())
    return result.hexdigest()
