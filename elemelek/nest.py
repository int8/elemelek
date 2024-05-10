import os
import random
import shutil
from functools import cached_property
from typing import List, Callable, Optional

import jsonlines
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
from elemelek.utils import calculate_file_md5, get_samples_per_group
from transformers import AutoTokenizer


class Elemelek(SelfLogging):
    def __init__(self, config: Egg):
        self.config = config
        self.__dataset_id = calculate_file_md5(config.dataset_jsonl_path)
        os.makedirs(self.dataset_artifacts_path, exist_ok=True)

        self.db = InstructionsDB(self.dataset_artifacts_path)
        self.index = InstructionsSemanticIndex(self.index_dir_path)
        self._clustering = None
        if not os.path.exists(self.config_path):
            shutil.copy(config.path, self.config_path)
            self._build()

    @property
    def config_path(self):
        return os.path.join(ELEMELEK_ARTIFACTS_PATH, self.__dataset_id, "config.yaml")

    @property
    def dataset_artifacts_path(self):
        return os.path.join(ELEMELEK_ARTIFACTS_PATH, self.__dataset_id)

    @property
    def index_dir_path(self):
        return os.path.join(self.dataset_artifacts_path, "index")

    @property
    def clustering(self) -> list[InstructionsCluster]:
        return self.index.cluster(k=self.config.semantic_index.n_clusters)

    @cached_property
    def feature_names(self) -> List[str]:
        return self.db.list_feature_names()

    @staticmethod
    def list_datasets():
        datasets = dict()
        for dataset_id in os.listdir(ELEMELEK_ARTIFACTS_PATH):
            config = Egg.from_yaml(
                os.path.join(ELEMELEK_ARTIFACTS_PATH, dataset_id, "config.yaml")
            )
            datasets[dataset_id] = config
        return datasets

    @staticmethod
    def remove_dataset(dataset_id: str):
        if os.path.exists(os.path.join(ELEMELEK_ARTIFACTS_PATH, dataset_id)):
            shutil.rmtree(os.path.join(ELEMELEK_ARTIFACTS_PATH, dataset_id))

    def _build(self):
        try:
            self.db.load_jsonl(
                self.config.dataset_jsonl_path,
                chunksize=self.config.db.database_insert_batch_size,
            )
            self.index.build(
                self.db,
                model_name=self.config.semantic_index.embeddings_model_name,
                batch_size=self.config.semantic_index.embeddings_computation_batch_size,
                dtype=self.config.semantic_index.dtype,
                metric=self.config.semantic_index.metric,
                connectivity=self.config.semantic_index.connectivity,
                expansion_add=self.config.semantic_index.expansion_add,
                expansion_search=self.config.semantic_index.expansion_search,
            )

            self.index.cluster(k=self.config.semantic_index.n_clusters)
            if self.config.features.basic:
                self.extract_basic_features()
            if self.config.features.reranker:
                self.extract_rerank_relevancy(
                    reranker_model_name=self.config.features.reranker.model_name,
                    reranker_batch_size=self.config.features.reranker.batch_size,
                )
            if self.config.features.language_tool:
                self.extract_lang_mistakes(
                    lang=self.config.features.language_tool.lang,
                    nr_of_threads=self.config.features.language_tool.nr_of_threads,
                )
        except Exception as e:
            Elemelek.remove_dataset(self.__dataset_id)
            raise e

    @classmethod
    def from_dataset_id(cls, dataset_id: str):
        if dataset_id in Elemelek.list_datasets():
            config = Egg.from_yaml(
                os.path.join(ELEMELEK_ARTIFACTS_PATH, dataset_id, "config.yaml")
            )
            return cls(config)

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

    def to_pandas(
        self, indices: Optional[List[int]] = None, include_features: bool = True
    ):
        if indices:
            return pd.DataFrame(
                [
                    elem.to_dict(include_features=include_features)
                    for elem in self.db.yield_subset(indices)
                ]
            )
        return pd.DataFrame([elem.to_flat_dict() for elem in self.db])

    def start_sampling(self, shuffle: bool = False):
        return ElemelekSample(self, self.db.ids, shuffle=shuffle)

    def to_jsonl(
        self,
        output_file_path: str,
        indices: Optional[List[int]] = None,
        include_features: bool = False,
        use_chat_template_from: Optional[str] = "Facebook/blenderbot-400M-distill",
    ):
        tokenizer = None
        if use_chat_template_from:
            tokenizer = AutoTokenizer.from_pretrained(use_chat_template_from)
        with jsonlines.open(output_file_path, "w") as jsonl_f:

            def __process_instruction(e):
                elem_dict = e.to_dict(include_features=include_features)
                if tokenizer:
                    __instruction_text = tokenizer.apply_chat_template(
                        e.to_conversation_dict(), tokenize=False
                    )
                    elem_dict |= {"__instruction_text": __instruction_text}
                jsonl_f.write(elem_dict)

            if indices:
                for elem in self.db.yield_subset(indices):
                    __process_instruction(elem)
            else:
                for elem in self.db:
                    __process_instruction(elem)


class ElemelekSample(SelfLogging):
    def __init__(self, elemelek: Elemelek, ids: List[int], shuffle: bool = False):
        self.elemelek = elemelek
        self.ids = ids
        if shuffle:
            self.shuffle()

    def shuffle(self) -> "ElemelekSample":
        return ElemelekSample(
            elemelek=self.elemelek, ids=random.sample(self.ids, len(self.ids))
        )

    def unique(self) -> "ElemelekSample":
        return ElemelekSample(elemelek=self.elemelek, ids=self.elemelek.db.unique_ids())

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

        samples_per_cluster = get_samples_per_group(clustering, k)

        if method == SubsetChoiceMethod.RANDOM:
            for i, c in tqdm(enumerate(clustering)):
                return_ids += c.random_sample(samples_per_cluster[i])

        if method == SubsetChoiceMethod.VARIABILITY_FACTOR:
            within_cluster_diversity_factor = kwargs.get(
                "within_cluster_diversity_factor"
            )
            if within_cluster_diversity_factor is None:
                raise ValueError(
                    f"Please provide float within_cluster_diversity_factor "
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
                    within_cluster_diversity_factor=within_cluster_diversity_factor,
                )

                assert len(return_ids) == sum(samples_per_cluster[: (i + 1)])

        return ElemelekSample(self.elemelek, return_ids)

    def stratify(self, feature_name: str, k: int) -> "ElemelekSample":
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
        groups = [g[1].index.tolist() for g in s.groupby(s)]
        samples_per_group = get_samples_per_group(groups, k)
        sample_ids = []
        for i, sample_size in enumerate(samples_per_group):
            sample_ids += random.sample(groups[i], sample_size)

        return ElemelekSample(self.elemelek, sample_ids)

    def sample_random(self, k: int) -> "ElemelekSample":
        return ElemelekSample(elemelek=self.elemelek, ids=random.sample(self.ids, k=k))

    def __len__(self):
        return len(self.ids)

    def __add__(self, other):
        return ElemelekSample(
            elemelek=self.elemelek, ids=list(set(self.ids) | set(other.ids))
        )

    def to_pandas(self, include_features: bool = True):
        self.elemelek.to_pandas(self.ids, include_features=include_features)

    def to_jsonl(
        self,
        output_file_path: str,
        include_features: bool = False,
        use_chat_template_from: Optional[str] = "Facebook/blenderbot-400M-distill",
    ):
        self.elemelek.to_jsonl(
            output_file_path=output_file_path,
            indices=self.ids,
            include_features=include_features,
            use_chat_template_from=use_chat_template_from,
        )
