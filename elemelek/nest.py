import os
import random
import shutil
from typing import List, Callable, Optional

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
from elemelek.utils import calculate_file_md5, stratified_sample


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

    def to_pandas(self, indices: Optional[List[int]] = None):
        if indices:
            return pd.DataFrame(
                [elem.to_flat_dict() for elem in self.db.yield_subset(indices)]
            )
        return pd.DataFrame([elem.to_flat_dict() for elem in self.db])

    def as_sample(self, randomize: bool = False):
        return ElemelekSample(self, self.db.ids, shuffle=randomize)


class ElemelekSample(SelfLogging):
    def __init__(self, elemelek: Elemelek, ids: List[int], shuffle: bool = False):
        self.elemelek = elemelek
        self.ids = ids
        if shuffle:
            self.shuffle()

    def shuffle(self):
        random.shuffle(self.ids)

    def filter(
        self, f: Callable[[Instruction], bool], max_k: Optional[int] = None
    ) -> "ElemelekSample":
        filtered_ids = list()
        for instruction in tqdm(
            self.elemelek.db.yield_subset(self.ids),
            desc="Filtering dataset...",
            total=len(self.ids),
        ):
            if f(instruction):
                filtered_ids.append(instruction.id)
            if max_k:
                filtered_ids = filtered_ids[:max_k]
        return ElemelekSample(self.elemelek, filtered_ids)

    def sample_diverse(
        self, k: int, method: SubsetChoiceMethod, **kwargs
    ) -> "ElemelekSample":
        if len(self.ids) <= k:
            self.info(f"Your current subset has less than {k} elements")
            return self
        return_ids = []
        ids_set = set(self.ids)
        clustering = [
            cluster.keep(ids_set)
            for cluster in tqdm(
                self.elemelek.clustering,
                "Filtering clustering to keep sample elements only ...",
            )
        ]
        clustering = [c for c in clustering if len(c) > 0]

        samples_per_cluster = InstructionsCluster.get_samples_per_cluster(clustering, k)

        if method == SubsetChoiceMethod.RANDOM:
            for i, c in tqdm(enumerate(clustering)):
                return_ids += c.random_sample(samples_per_cluster[i])

        if method == SubsetChoiceMethod.TARGET_MEDIAN_DISTANCE:
            target_median = kwargs.get("target_median")
            if target_median is None:
                raise ValueError(
                    f"Please provide float target_median "
                    f"parameter when  choosing {method}"
                )
            for i, c in tqdm(
                enumerate(clustering),
                total=len(clustering),
                desc="Solving diverse subset problem heuristically "
                "using genetic algorithms",
            ):
                if samples_per_cluster[i] == 0:
                    continue
                # TODO: multiprocessing approach would help here
                return_ids += c.get_semantically_similar_sample(
                    index=self.elemelek.index,
                    k=samples_per_cluster[i],
                    target_distance_median=target_median,
                )

                assert len(return_ids) == sum(samples_per_cluster[: (i + 1)])

        return ElemelekSample(self.elemelek, return_ids)

    def sample_uniform_categorical(self, feature_name: str, k: int) -> "ElemelekSample":
        if len(self.ids) <= k:
            self.info(f"Your current subset has less than {k} elements")
            return self
        values = []
        indices = []
        for instruction in tqdm(
            self.elemelek.db.yield_subset(self.ids),
            desc=f"retrieving values of {feature_name}",
            total=len(self.ids),
        ):
            values.append(instruction.get_feature(feature_name).value)
            indices.append(instruction.id)
        s = pd.Series(values, index=indices)
        sampled_ids = stratified_sample(s, k).index.tolist()
        return ElemelekSample(self.elemelek, sampled_ids)

    def sample_random(self, k: int) -> "ElemelekSample":
        return ElemelekSample(elemelek=self.elemelek, ids=random.sample(self.ids, k=k))

    def __len__(self):
        return len(self.ids)

    def __add__(self, other):
        return ElemelekSample(
            elemelek=self.elemelek, ids=list(set(self.ids) | set(other.ids))
        )

    def to_pandas(self):
        self.elemelek.to_pandas(self.ids)
