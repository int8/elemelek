from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np
from sentence_transformers import CrossEncoder
from torch import nn
from tqdm import tqdm

from elemelek.logging import SelfLogging
from elemelek.model import (
    Instruction,
    InstructionFeature,
    CustomExtractorDefinition,
)


from elemelek.utils import divide_dict, language_tool_scan
import concurrent.futures


RERANKER_RELEVANCE_SCORE = "reranker-relevance-score"
LANGUAGE_TOOL_CHECK = "language_tool_check"
MEDIAN_WORD_LENGTH = "median_word_length"
QUANTILE_WORD_LENGTH_09 = "quantile_0.9_word_length"
QUANTILE_WORD_LENGTH_01 = "quantile_0.1_word_length"
TOTAL_LENGTH = "total_length"
IS_QUESTION = "is_question"
HAS_INPUT = "has_input"
NUMERIC_CHARS_RATIO = "numeric_chars_ratio"
NON_ALPHA_NUMERIC_CHARS_RATIO = "non_alpha_numeric_chars_ratio"


class InstructionFeatureExtractor(ABC, SelfLogging):
    @abstractmethod
    def extract(self, instruction: Instruction) -> List[InstructionFeature]:
        pass

    def extract_many(
        self, instructions: Sequence[Instruction]
    ) -> List[InstructionFeature]:
        features = []
        for instruction in tqdm(
            instructions, desc=f"Extracting features via {self.__class__.__name__}..."
        ):
            features += self.extract(instruction)

        return features


class BasicFeaturesExtractor(InstructionFeatureExtractor):
    def extract(self, instruction: Instruction) -> List[InstructionFeature]:
        words = instruction.text.split()
        word_lengths = [len(word) for word in words]
        non_alpha_num_chars = sum(1 for char in instruction.text if not char.isalnum())
        numeric_chars = sum(1 for char in instruction.text if char.isdigit())

        return [
            InstructionFeature(
                name=MEDIAN_WORD_LENGTH,
                value=np.median(word_lengths),
                instruction_id=instruction.id,
            ),
            InstructionFeature(
                name=QUANTILE_WORD_LENGTH_09,
                value=np.quantile(word_lengths, 0.9),
                instruction_id=instruction.id,
            ),
            InstructionFeature(
                name=QUANTILE_WORD_LENGTH_01,
                value=np.quantile(word_lengths, 0.1),
                instruction_id=instruction.id,
            ),
            InstructionFeature(
                name=TOTAL_LENGTH,
                value=len(instruction.text),
                instruction_id=instruction.id,
            ),
            InstructionFeature(
                name=IS_QUESTION,
                value=instruction.is_question(),
                instruction_id=instruction.id,
            ),
            InstructionFeature(
                name=HAS_INPUT,
                value=instruction.input is not None,
                instruction_id=instruction.id,
            ),
            InstructionFeature(
                name=NUMERIC_CHARS_RATIO,
                value=numeric_chars / len(instruction.text),
                instruction_id=instruction.id,
            ),
            InstructionFeature(
                name=NON_ALPHA_NUMERIC_CHARS_RATIO,
                value=non_alpha_num_chars / len(instruction.text),
                instruction_id=instruction.id,
            ),
        ]


# https://stackoverflow.com/a/73464315
class MultiThreadedLanguageToolFeatureExtractor(InstructionFeatureExtractor):
    def __init__(self, lang_code: str = "pl-PL", nr_of_threads: int = 4):
        self.lang_code = lang_code
        self.nr_of_threads = nr_of_threads

    def extract(self, instruction: Instruction) -> InstructionFeature:
        raise NotImplemented(
            "single instruction extract is not supported in multithreaded setup"
        )

    def extract_many(
        self, instructions: Sequence[Instruction]
    ) -> List[InstructionFeature]:
        final_results = {}
        instructions_dict = {
            instruction.id: instruction for instruction in instructions
        }  # This bit sucks - it loads everything into memory 😬💀
        chunks = divide_dict(instructions_dict, self.nr_of_threads)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.nr_of_threads
        ) as executor:
            futures = [
                executor.submit(language_tool_scan, instructions_dict, chunk)
                for chunk in chunks
            ]
        for future in concurrent.futures.as_completed(futures):
            final_results.update(future.result())

        return [final_results[instruction.id] for instruction in instructions]


class CustomFeatureExtractor(InstructionFeatureExtractor):
    def __init__(self, custom_extractors: List[CustomExtractorDefinition]):
        self.customer_extractors = custom_extractors

    def extract(self, instruction: Instruction) -> List[InstructionFeature]:
        return [extractor.f(instruction) for extractor in self.customer_extractors]


class RerankerRelevanceScoreFeatureExtractor(InstructionFeatureExtractor):
    def __init__(self, reranker: CrossEncoder, batch_size: int = 16):
        self.reranker = reranker
        self.batch_size = batch_size

    def extract(self, instruction: Instruction) -> InstructionFeature:
        results = self.reranker.predict(
            [[instruction.qa.question, instruction.qa.answer]],
            activation_fct=nn.Sigmoid(),
        )
        return InstructionFeature(
            instruction_id=instruction.id,
            name=RERANKER_RELEVANCE_SCORE,
            value=float(results[0]),
        )

    def extract_many(
        self, instructions: Sequence[Instruction]
    ) -> List[InstructionFeature]:
        results = []
        for start_index in tqdm(
            list(range(0, len(instructions), self.batch_size)),
            desc=f"Computing rerank relevance score via {self.reranker.model.config._name_or_path}",
        ):
            end_index = start_index + self.batch_size
            batch = instructions[start_index:end_index]
            batch_results = self.reranker.predict(
                [
                    [instruction.qa.question, instruction.qa.answer]
                    for instruction in batch
                ],
                activation_fct=nn.Sigmoid(),
            )

            results += [
                InstructionFeature(
                    name=RERANKER_RELEVANCE_SCORE,
                    value=float(batch_results[i]),
                    instruction_id=b.id,
                )
                for i, b in enumerate(batch)
            ]

        return results
