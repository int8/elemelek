import hashlib

from collections import Counter
from typing import List, Dict

import language_tool_python
import numpy as np
import pandas as pd

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


def stratified_sample(series: pd.Series, total_samples: int):
    value_counts = series.value_counts()
    num_categories = len(value_counts)
    samples_per_cat = total_samples // num_categories
    samples_remaining = total_samples % num_categories

    possible_sample_sizes = np.minimum(value_counts, samples_per_cat).to_dict()

    for category in value_counts.index:
        if (
            samples_remaining > 0
            and possible_sample_sizes[category] < value_counts[category]
        ):
            possible_sample_sizes[category] += 1
            samples_remaining -= 1

    sampled_data = series.groupby(series).apply(
        lambda x: x.sample(n=possible_sample_sizes[x.name], random_state=1)
    )
    sampled_data.index = sampled_data.index.droplevel(0)

    return sampled_data
