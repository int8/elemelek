import dataclasses
import json
import math
import os
import random
import shutil
from functools import cached_property
from typing import List, Sequence, Set


import numpy as np
from sentence_transformers import SentenceTransformer

from usearch.index import Index

from elemelek.genetic import OptimalSubsetGeneticAlgorithm
from elemelek.logging import SelfLogging
from elemelek.model import (
    EmbeddingComputationStrategy,
    Instruction,
)
from elemelek.utils import calculate_text_chunk_md5


from tqdm import tqdm


@dataclasses.dataclass
class InstructionsCluster:
    centroid_id: int
    elements_ids: List[int]

    def dict(self):
        return dataclasses.asdict(self)

    def keep(self, ids: Set[int]) -> "InstructionsCluster":
        return InstructionsCluster(
            centroid_id=self.centroid_id,
            elements_ids=[e for e in self.elements_ids if e in ids],
        )

    def random_sample(self, n: int):
        return random.sample(self.elements_ids, n)

    def get_similarity_matrix(self, index: "InstructionsSemanticIndex"):
        similarity_matrix = np.zeros(
            shape=(len(self.elements_ids), len(self.elements_ids))
        )
        for i, idx_i in enumerate(self.elements_ids):
            for j, idx_j in enumerate(self.elements_ids):
                if i > j:
                    continue
                similarity_matrix[i, j] = index.index.pairwise_distance(idx_i, idx_j)
                similarity_matrix[j, i] = similarity_matrix[idx_i, idx_j]
        return similarity_matrix

    def get_semantically_similar_sample(
        self,
        index: "InstructionsSemanticIndex",
        k: int,
        target_similarity_median: float,
    ) -> "InstructionsCluster":
        similarity_matrix = self.get_similarity_matrix(index)
        ga = OptimalSubsetGeneticAlgorithm(
            similarity_matrix,
            target_similarity_median,
            population_size=100,
            sample_size=k - 1,
            generations=25,
            mutation_rate=0.1,
        )
        solution = ga.optimize()
        return InstructionsCluster(
            centroid_id=self.centroid_id, elements_ids=solution + [self.centroid_id]
        )

    def __len__(self):
        return len(self.elements_ids)

    @staticmethod
    def get_samples_per_cluster(
        clustering: List["InstructionsCluster"], k: int
    ) -> List[int]:
        num_clusters = len(clustering)
        ideal_samples_per_cluster = math.floor(k / num_clusters)

        num_per_cluster = [
            min(len(cluster), ideal_samples_per_cluster) for cluster in clustering
        ]

        total_samples_from_uniform = sum(num_per_cluster)

        remaining_samples = k - total_samples_from_uniform

        while remaining_samples > 0:
            for i, cluster in enumerate(clustering):
                if num_per_cluster[i] < len(cluster) and remaining_samples > 0:
                    num_per_cluster[i] += 1
                    remaining_samples -= 1
                if remaining_samples <= 0:
                    break

        return num_per_cluster


class InstructionsSemanticIndex(SelfLogging):
    def __init__(self, index_dir_path: str):
        self.index_dir_path = index_dir_path
        os.makedirs(index_dir_path, exist_ok=True)
        os.makedirs(self.embeddings_path, exist_ok=True)

        self.__model_name = None
        self.index = None

        if os.path.exists(self.index_path):
            self.info(f"Index at {self.index_path} already exists - loading it")
            self.index = Index.restore(self.index_path)

        if os.path.exists(self.encoder_config_path):
            with open(self.encoder_config_path, "r") as f:
                encoder_data = json.load(f)
                self.__model_name = encoder_data["model_name"]

    @cached_property
    def _encoder(self):
        self.debug(f"Loading {self.__model_name}")
        return SentenceTransformer(self.__model_name)

    @property
    def embeddings_path(self):
        return os.path.join(self.index_dir_path, "embeddings")

    @property
    def index_path(self):
        return os.path.join(self.index_dir_path, "index.bin")

    @property
    def encoder_config_path(self):
        return os.path.join(self.index_dir_path, "encoder.json")

    def build(
        self,
        instructions: Sequence[Instruction],
        model_name: str,
        batch_size: int = 1000,
        force: bool = False,
        **usearch_kwargs,
    ):
        self.__model_name = model_name
        if force or not self.index:
            self._compute_embeddings(
                instructions, EmbeddingComputationStrategy.TRUNCATE, batch_size
            )
            self.index = self._build_index(**usearch_kwargs)
        else:
            self.info(
                f"Index at {self.index_path} already loaded. "
                f"If you want to rebuild pass force = True"
            )

    def cluster(self, k: int = None) -> List[InstructionsCluster]:
        if not self.index:
            raise ValueError("Build  your index first via build()")

        if not k:
            k = len(self.index) // 100
            self.info(f"Nr of clusters not set - k automatically set to {k}")

        if os.path.exists(os.path.join(self.index_dir_path, f"clustering_{k}.json")):
            with open(
                os.path.join(self.index_dir_path, f"clustering_{k}.json"), "r"
            ) as f:
                self.debug(
                    f'Clustering {os.path.join(self.index_dir_path, f"clustering_{k}.json")} already exists'
                )
                clustering = json.load(f)
                return [InstructionsCluster(**c) for c in clustering]
        try:
            clustering = self.index.cluster(
                min_count=k,
                max_count=k,
            )
            centroids, _ = clustering.centroids_popularity
            clusters = []
            for centroid in centroids:
                cluster = InstructionsCluster(
                    centroid_id=int(centroid),
                    elements_ids=[
                        int(i) for i in clustering.members_of(centroid).tolist()
                    ],
                )
                clusters.append(cluster)
        except RuntimeError as e:
            if e.args[0] == "Index too small to cluster!":
                self.warn(
                    "Your index is to small to be clustered - "
                    "returning whole index as one cluster"
                )
                keys = [self.index.keys[k] for k in range(len(self.index))]
                clusters = [InstructionsCluster(centroid_id=keys[0], elements_ids=keys)]
            else:
                raise e
        with open(os.path.join(self.index_dir_path, f"clustering_{k}.json"), "w") as f:
            json.dump([c.dict() for c in clusters], f)
        return clusters

    def _build_index(self, save: bool = True, **usearch_kwargs):
        embeddings_dim = self._encoder.get_sentence_embedding_dimension()

        embeddings_files = os.listdir(self.embeddings_path)

        index = Index(ndim=embeddings_dim, **usearch_kwargs)
        for i, f in tqdm(
            enumerate(embeddings_files),
            total=len(embeddings_files),
            desc="Building HNSW index",
        ):
            data = np.load(os.path.join(self.embeddings_path, f))
            embeddings = data["embeddings"]
            keys = data["keys"]
            index.add(keys=[int(k) for k in keys], vectors=embeddings)

        if save:
            index.save(self.index_path)
            with open(self.encoder_config_path, "w") as f:
                json.dump({"model_name": self.__model_name}, f)
            self.info(f"Index saved at {self.index_path}")
        return index

    def _compute_embeddings(
        self,
        instructions: Sequence[Instruction],
        strategy: EmbeddingComputationStrategy,
        batch_size: int = 1000,
    ):
        if strategy == EmbeddingComputationStrategy.TRUNCATE:
            self._compute_truncated_embeddings(instructions, batch_size)

    def _compute_truncated_embeddings(
        self, instructions: Sequence[Instruction], batch_size: int = 3
    ):
        shutil.rmtree(self.embeddings_path)
        os.makedirs(self.embeddings_path, exist_ok=True)
        embedder = self._encoder
        for start_index in tqdm(
            list(range(0, len(instructions), batch_size)), desc="Computing embeddings"
        ):
            end_index = start_index + batch_size
            batch = instructions[start_index:end_index]
            collection_hash = calculate_text_chunk_md5([b.text for b in batch])
            if os.path.exists(
                os.path.join(self.embeddings_path, f"{collection_hash}.bin")
            ):
                continue
            keys = [b.id for b in batch]
            embeddings = embedder.encode([b.text for b in batch])

            with open(
                os.path.join(self.embeddings_path, f"{collection_hash}.bin"), "wb"
            ) as fp:
                np.savez_compressed(fp, keys=keys, embeddings=embeddings)

    def search(self, query: str, k: int = 2) -> List[int]:
        encoded_query = self._encoder.encode(query)
        matches = self.index.search(encoded_query, count=k)
        return [m.key for m in matches]
