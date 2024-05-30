import dataclasses
import os
from distutils.util import strtobool

import yaml
from dotenv import load_dotenv

load_dotenv()
LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "DEBUG")
LOGGING_FORMAT = os.environ.get(
    "LOGGING_FORMAT", "[%(asctime)s] [%(levelname)8s] --- %(name)s - %(message)s"
)
LOGGING_PLAIN = strtobool(os.environ.get("LOGGING_PLAIN", "false"))
from pathlib import Path


ELEMELEK_ARTIFACTS_PATH = os.getenv(
    "ELEMELEK_ARTIFACTS_PATH", os.path.join(str(Path.home()), ".elemelek")
)

if not os.path.exists(ELEMELEK_ARTIFACTS_PATH):
    os.makedirs(ELEMELEK_ARTIFACTS_PATH, exist_ok=True)

INSTRUCTION_FIELD = "instruction"
INPUT_FIELD = "input"
OUTPUT_FIELD = "output"
MANDATORY_FIELDS = {INSTRUCTION_FIELD, INPUT_FIELD, OUTPUT_FIELD}


@dataclasses.dataclass
class DBConfig:
    database_insert_batch_size: int
    remove_duplicates: bool


@dataclasses.dataclass
class SemanticIndexConfig:
    embeddings_model_name: str
    embeddings_computation_batch_size: int
    metric: str
    connectivity: int
    dtype: str
    expansion_add: int
    expansion_search: int
    n_clusters: int


@dataclasses.dataclass
class RerankerConfig:
    model_name: str
    batch_size: int
    strategy: str


@dataclasses.dataclass
class LanguageToolConfig:
    nr_of_threads: int
    lang: str


@dataclasses.dataclass
class FeaturesConfig:
    basic: bool
    reranker: RerankerConfig
    language_tool: LanguageToolConfig


@dataclasses.dataclass
class Egg:
    path: str
    dataset_jsonl_path: str
    db: DBConfig
    semantic_index: SemanticIndexConfig
    features: FeaturesConfig

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as file:
            config_data = yaml.safe_load(file)
        return cls(
            path=path,
            dataset_jsonl_path=config_data["dataset_jsonl_path"],
            db=DBConfig(**config_data["db"]),
            semantic_index=SemanticIndexConfig(**config_data["semantic_index"]),
            features=FeaturesConfig(
                basic=config_data["features"]["basic"],
                reranker=RerankerConfig(**config_data["features"]["reranker"])
                if config_data["features"]["reranker"]
                else None,
                language_tool=LanguageToolConfig(
                    **config_data["features"]["language_tool"]
                )
                if config_data["features"]["language_tool"]
                else None,
            ),
        )
