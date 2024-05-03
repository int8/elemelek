import dataclasses
import enum
from functools import cached_property
from typing import Optional, Any, List, Callable


@dataclasses.dataclass
class InstructionFeature:
    instruction_id: int
    name: str
    value: Any


@dataclasses.dataclass
class Question:
    question: str
    answer: str


@dataclasses.dataclass
class Instruction:
    id: int
    instruction: str
    output: str
    input: Optional[str]
    features: Optional[List[InstructionFeature]] = dataclasses.field(
        default=None,
    )

    @cached_property
    def text(self):
        return (
            self.instruction
            + "\n"
            + (self.input if self.input else "")
            + " \n"
            + self.output
        )

    @property
    def qa(self) -> Optional[Question]:
        question = self.instruction + ((" \n" + self.input) if self.input else "")
        answer = self.output

        return Question(question=question, answer=answer)

    def is_question(self) -> bool:
        return self.instruction.strip().endswith("?")

    def to_flat_dict(self):
        return {
            "id": self.id,
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        } | {
            f"feature_{feature.name}": feature.value
            for feature in self.features or dict()
        }


@dataclasses.dataclass
class CustomExtractorDefinition:
    name: str
    f: Callable[[Instruction], InstructionFeature]


class SubsetChoiceMethod(enum):
    RANDOM = "RANDOM"
    TARGET_MEDIAN_SIMILARITY = "TARGET_MEDIAN_SIMILARITY"


class EmbeddingComputationStrategy(enum.Enum):
    TRUNCATE = "TRUNCATE"


class DatasetError(BaseException):
    pass
