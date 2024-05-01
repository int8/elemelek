import os
import shutil
from typing import List

import pandas as pd
import torch
from sentence_transformers import CrossEncoder

from elemelek.db import InstructionsDB
from elemelek.features import (
    BasicFeaturesExtractor,
    RerankerRelevanceScoreFeatureExtractor,
    MultiThreadedLanguageToolFeatureExtractor,
)
from elemelek.index import InstructionsSemanticIndex
from elemelek.logging import SelfLogging
from elemelek.model import Instruction
from elemelek.settings import ELEMELEK_ARTIFACTS_PATH, ElemelekConfig
from elemelek.utils import calculate_file_md5


class Elemelek(SelfLogging):
    def __init__(
        self,
        dataset_id: str,
    ):
        self.__dataset_id = dataset_id
        os.makedirs(self.dataset_artifacts_path, exist_ok=True)
        self._db = InstructionsDB(self.dataset_artifacts_path)
        self._index = InstructionsSemanticIndex(self.index_dir_path)

    @property
    def dataset_artifacts_path(self):
        return os.path.join(ELEMELEK_ARTIFACTS_PATH, self.__dataset_id)

    @property
    def config_path(self):
        return os.path.join(self.dataset_artifacts_path, "config.yaml")

    @classmethod
    def from_config(cls, config: ElemelekConfig):
        dataset_id = calculate_file_md5(config.dataset_jsonl_path)
        try:
            if dataset_id in Elemelek.list_datasets():
                return cls(dataset_id)

            elemelek = cls(dataset_id)
            elemelek._db.load_jsonl(config.dataset_jsonl_path)
            elemelek._index.build(
                elemelek._db,
                model_name=config.semantic_index.embeddings_model_name,
                batch_size=config.semantic_index.embeddings_computation_batch_size,
            )
            elemelek._index.cluster(k=config.semantic_index.n_clusters)
            if config.features.basic:
                elemelek.extract_basic_features()
            if config.features.reranker:
                elemelek.extract_rerank_relevancy(
                    reranker_model_name=config.features.reranker.model_name,
                    reranker_batch_size=config.features.reranker.batch_size,
                )
            if config.features.language_tool:
                elemelek.extract_lang_mistakes(
                    lang=config.features.language_tool.lang,
                    nr_of_threads=config.features.language_tool.nr_of_threads,
                )
            return elemelek
        except Exception as e:
            Elemelek.remove_dataset(dataset_id)
            raise e

    @staticmethod
    def list_datasets():
        return os.listdir(ELEMELEK_ARTIFACTS_PATH)

    @staticmethod
    def remove_dataset(dataset_id: str):
        if os.path.exists(os.path.join(ELEMELEK_ARTIFACTS_PATH, dataset_id)):
            shutil.rmtree(os.path.join(ELEMELEK_ARTIFACTS_PATH, dataset_id))

    @property
    def index_dir_path(self):
        return os.path.join(self.dataset_artifacts_path, "index")

    def extract_basic_features(self):
        extractor = BasicFeaturesExtractor()
        features = extractor.extract_many(self._db)
        self._db.insert_features(features)

    def extract_rerank_relevancy(
        self, reranker_model_name: str, reranker_batch_size: int
    ):
        encoder = CrossEncoder(
            reranker_model_name,
            default_activation_function=torch.nn.Sigmoid(),
            max_length=512,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        extractor = RerankerRelevanceScoreFeatureExtractor(
            encoder, batch_size=reranker_batch_size
        )
        features = extractor.extract_many(self._db)
        self._db.insert_features(features)

    def extract_lang_mistakes(self, lang: str, nr_of_threads: int):
        extractor = MultiThreadedLanguageToolFeatureExtractor(
            lang_code=lang, nr_of_threads=nr_of_threads
        )
        features = extractor.extract_many(self._db)
        self._db.insert_features(features)

    def search(self, query: str, k: int = 10) -> List[Instruction]:
        instruction_ids = self._index.search(query, k)
        return [self._db[int(idx)] for idx in instruction_ids]

    def to_pandas(self):
        return pd.DataFrame([elem.to_flat_dict() for elem in self._db])
