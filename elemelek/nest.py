import os
import random
import shutil
from typing import List, Callable

import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from elemelek.db import InstructionsDB
from elemelek.features import (
    BasicFeaturesExtractor,
    RerankerRelevanceScoreFeatureExtractor,
    MultiThreadedLanguageToolFeatureExtractor,
)
from elemelek.index import InstructionsSemanticIndex, InstructionsCluster
from elemelek.logging import SelfLogging
from elemelek.model import Instruction, SubsetChoiceMethod
from elemelek.settings import ELEMELEK_ARTIFACTS_PATH, Egg
from elemelek.utils import calculate_file_md5


class InstructionsClustering:
    pass


class Elemelek(SelfLogging):
    def __init__(self, dataset_id: str, config: Egg):
        self.__dataset_id = dataset_id
        os.makedirs(self.dataset_artifacts_path, exist_ok=True)

        self.db = InstructionsDB(self.dataset_artifacts_path)
        self.index = InstructionsSemanticIndex(self.index_dir_path)
        self.config = config
        self._clustering = None

    @property
    def dataset_artifacts_path(self):
        return os.path.join(ELEMELEK_ARTIFACTS_PATH, self.__dataset_id)

    @property
    def index_dir_path(self):
        return os.path.join(self.dataset_artifacts_path, "index")

    @property
    def clustering(self) -> list[InstructionsCluster]:
        return self.index.cluster(k=self.config.semantic_index.n_clusters)

    @staticmethod
    def list_datasets():
        return os.listdir(ELEMELEK_ARTIFACTS_PATH)

    @staticmethod
    def remove_dataset(dataset_id: str):
        if os.path.exists(os.path.join(ELEMELEK_ARTIFACTS_PATH, dataset_id)):
            shutil.rmtree(os.path.join(ELEMELEK_ARTIFACTS_PATH, dataset_id))

    @classmethod
    def hatch(cls, config: Egg):
        dataset_id = calculate_file_md5(config.dataset_jsonl_path)
        try:
            if dataset_id in Elemelek.list_datasets():
                return cls(dataset_id, config=config)

            elemelek = cls(dataset_id, config=config)
            elemelek.db.load_jsonl(config.dataset_jsonl_path)
            elemelek.index.build(
                elemelek.db,
                model_name=config.semantic_index.embeddings_model_name,
                batch_size=config.semantic_index.embeddings_computation_batch_size,
                dtype=config.semantic_index.dtype,
                metric=config.semantic_index.metric,
                connectivity=config.semantic_index.connectivity,
                expansion_add=config.semantic_index.expansion_add,
                expansion_search=config.semantic_index.expansion_search,
            )

            elemelek.index.cluster(k=config.semantic_index.n_clusters)
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

    def extract_basic_features(self):
        extractor = BasicFeaturesExtractor()
        features = extractor.extract_many(self.db)
        self.db.insert_features(features)

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
        features = extractor.extract_many(self.db)
        self.db.insert_features(features)

    def extract_lang_mistakes(self, lang: str, nr_of_threads: int):
        extractor = MultiThreadedLanguageToolFeatureExtractor(
            lang_code=lang, nr_of_threads=nr_of_threads
        )
        features = extractor.extract_many(self.db)
        self.db.insert_features(features)

    def search(self, query: str, k: int = 10) -> List[Instruction]:
        instruction_ids = self.index.search(query, k)
        return [self.db[int(idx)] for idx in instruction_ids]

    def to_pandas(self):
        return pd.DataFrame([elem.to_flat_dict() for elem in self.db])

    def as_sample(self, randomize: bool = False):
        return ElemelekSample(self, self.db.ids, randomize=randomize)


class ElemelekSample(SelfLogging):
    def __init__(self, elemelek: Elemelek, ids: List[int], randomize: bool = False):
        self.elemelek = elemelek
        if randomize:
            random.shuffle(ids)
        self.ids = ids

    def filter(self, f: Callable[[Instruction], bool]) -> "ElemelekSample":
        filtered_ids = list()
        for instruction in tqdm(
            self.elemelek.db[self.ids],
            desc="Filtering dataset...",
            total=len(self.ids),
        ):
            if f(instruction):
                filtered_ids.append(instruction.id)
        return ElemelekSample(self.elemelek, filtered_ids)

    def get_diverse_subset(
        self, k: int, method: SubsetChoiceMethod, **kwargs
    ) -> "ElemelekSample":
        if len(self.ids) <= k:
            self.info(f"Your current subset has less than {k} elements")
            return self
        self.info("X")
        return_ids = []
        ids_set = set(self.ids)
        clustering = [cluster.keep(ids_set) for cluster in self.elemelek.clustering]
        self.info(clustering)
        clustering = [c for c in clustering if len(c) > 0]
        samples_per_cluster = InstructionsCluster.get_samples_per_cluster(clustering, k)
        self.info(samples_per_cluster)
        if method == SubsetChoiceMethod.RANDOM:
            for i, c in tqdm(enumerate(clustering)):
                return_ids += c.random_sample(samples_per_cluster[i])

        if method == SubsetChoiceMethod.TARGET_MEDIAN_SIMILARITY:
            target_median = kwargs.get("target_median")
            if target_median is None:
                raise ValueError(
                    f"Please provide float target_median "
                    f"parameter when  choosing {method}"
                )
            for i, c in enumerate(clustering):
                return_ids += c.get_semantically_similar_sample(
                    index=self.elemelek.index,
                    k=samples_per_cluster[i],
                    target_similarity_median=target_median,
                )

        return ElemelekSample(self.elemelek, return_ids)

    def sample_uniform_categorical(self, feature_name: str, k: int) -> "ElemelekSample":
        if len(self.ids) <= k:
            self.info(f"Your current subset has less than {k} elements")
            return self
        values = []
        indices = []
        for instruction in tqdm(
            self.elemelek.db[self.ids],
            desc=f"retrieving values of {feature_name}",
            total=len(self.ids),
        ):
            values.append(instruction.get_feature(feature_name).value)
            indices.append(instruction.id)
        s = pd.Series(values, index=indices)
        category_counts = s.value_counts(normalize=True)
        weights = 1 / category_counts[s]
        weights = weights / weights.sum()
        sampled_ids = s.sample(
            n=k, weights=pd.Series(weights.values, index=s.index)
        ).index.values.tolist()

        return ElemelekSample(self.elemelek, sampled_ids)
