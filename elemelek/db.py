import json
import os
import sqlite3
import traceback
from typing import List, Dict, Callable, Any, Set

import pandas as pd
from tqdm import tqdm

from elemelek.logging import SelfLogging
from elemelek.model import DatasetError, Instruction, InstructionFeature
from elemelek.settings import MANDATORY_FIELDS
from elemelek.utils import count_jsonl_objects, compute_df_row_md5
from collections.abc import Sequence


class InMemoryFeatures(SelfLogging):
    def __init__(self, features_json_path: str):
        self.features_json_path = features_json_path
        self._feature_names = set()
        self._data = {}
        if os.path.exists(self.features_json_path):
            with open(self.features_json_path, "r") as f:
                _data = json.load(f)
                for k, v in _data.items():
                    self._data[int(k)] = v
                    self._feature_names |= set(v.keys())

        else:
            self._data: Dict[int, Dict[str, InstructionFeature]] = dict()

    def add(self, features: List[InstructionFeature], save: bool = True):
        for feature in features:
            self._data[feature.instruction_id] = self._data.get(
                feature.instruction_id, dict()
            ) | {feature.name: feature.value}
            self._feature_names |= {f.name for f in features}

        if save:
            self.save()

    def filter(self, feature_name, f: Callable[[Any], bool]) -> Set[int]:
        accepted_instuctions = set()
        for instruction_id in self._data:
            if f(self._data[instruction_id][feature_name]):
                accepted_instuctions.add(instruction_id)
        return accepted_instuctions

    def save(self):
        with open(self.features_json_path, "w") as f:
            json.dump(self._data, f)

    def get_features(self, instruction_id: int) -> List[InstructionFeature]:
        return [
            InstructionFeature(instruction_id=instruction_id, name=name, value=value)
            for name, value in self._data.get(instruction_id, dict()).items()
        ]

    def list_feature_names(self) -> List[str]:
        return list(self._feature_names)


class InstructionsDB(SelfLogging, Sequence):
    def __init__(self, data_dir_path: str):
        self.data_dir_path = data_dir_path
        self.conn = sqlite3.connect(self.sqlite_db_path)
        self.dataset_table_name = "dataset"
        self.features = InMemoryFeatures(self.features_json_path)

    @property
    def sqlite_db_path(self):
        return os.path.join(self.data_dir_path, "data.db")

    @property
    def features_json_path(self):
        return os.path.join(self.data_dir_path, "features.json")

    def close(self):
        self.conn.close()

    def load_jsonl(
        self, jsonl_path: str, chunksize: int = 25000, remove_duplicates: bool = True
    ):
        try:
            seen_instructions = set()
            reader = pd.read_json(jsonl_path, lines=True, chunksize=chunksize)
            duplicated = 0
            for chunk in tqdm(
                reader,
                total=count_jsonl_objects(jsonl_path) // chunksize,
                desc=f"Inserting chunks of {chunksize} entries to {self.sqlite_db_path}",
            ):
                if MANDATORY_FIELDS.difference(set(chunk.columns)):
                    raise DatasetError(
                        f"Make sure your json entries have all the fields"
                        f": {MANDATORY_FIELDS} "
                    )

                if remove_duplicates:
                    current_chunk_hashes = chunk[list[MANDATORY_FIELDS]].apply(
                        compute_df_row_md5, axis=1
                    )
                    keep_these = current_chunk_hashes.map(
                        lambda x: x not in seen_instructions
                    )
                    seen_instructions |= set(current_chunk_hashes.tolist())

                    duplicated += len(chunk) - keep_these.sum()
                    chunk = chunk[keep_these]
                self.info("1")
                chunk[list(MANDATORY_FIELDS)].to_sql(
                    self.dataset_table_name, self.conn, if_exists="append"
                )
                self.info("2")
                # other_columns = set(chunk.columns) - MANDATORY_FIELDS
                # other_features = []
                # if len(other_columns) > 0:
                #     for column in other_columns:
                #         other_features += [
                #             InstructionFeature(
                #                 name=column, instruction_id=id_, value=value
                #             )
                #             for id_, value in chunk[column].items()
                #         ]
                #     self.features.add(other_features, save=True)

            self.info(
                f"Data has been saved to database "
                + f"{duplicated} duplicates filtered out"
                if duplicated
                else ""
            )
        except Exception as e:
            self.error(traceback.format_exc())
            self.info(f"removing database {self.sqlite_db_path} due to error")
            os.remove(self.sqlite_db_path)
            raise e

    def __len__(self):
        cur = self.conn.cursor()
        res = cur.execute(f"SELECT count(*) from {self.dataset_table_name}")
        count = res.fetchone()
        return count[0]

    def __iter__(self):
        return self.__yield_all_rows()

    def __getitem__(self, idx: int | slice | list) -> Instruction | List[Instruction]:
        cur = self.conn.cursor()

        if isinstance(idx, slice) or isinstance(idx, list):
            if isinstance(idx, slice):
                indices = list(range(0, len(self)))[idx]
            else:
                indices = idx

            instructions = []
            for i in indices:
                instructions.append(self[i])
            return instructions
        elif isinstance(idx, int):
            res = cur.execute(
                f'SELECT "index", instruction, input, output FROM {self.dataset_table_name} WHERE "index" = {idx}'
            )
            row = res.fetchone()
            if row:
                id_, instruction, input_, output = row
                return Instruction(
                    id=id_,
                    instruction=instruction,
                    input=input_,
                    output=output,
                    features=self.features.get_features(instruction_id=idx),
                )

    def list_feature_names(self) -> List[str]:
        return self.features.list_feature_names()

    def insert_features(self, features: List[InstructionFeature]):
        self.features.add(features)

    def __yield_all_rows(self) -> Instruction:
        cursor = self.conn.cursor()
        cursor.execute(
            f'SELECT "index", instruction, input, output FROM {self.dataset_table_name}'
        )
        row = cursor.fetchone()
        while row:
            if row:
                id_, instruction, input_, output = row
                yield Instruction(
                    id=id_,
                    instruction=instruction,
                    input=input_,
                    output=output,
                    features=self.features.get_features(instruction_id=id_),
                )
            row = cursor.fetchone()
        cursor.close()

    def yield_subset(self, ids: List[int]):
        for id_ in ids:
            yield self[id_]

    @property
    def ids(self) -> List[int]:
        cursor = self.conn.cursor()
        res = cursor.execute(f'SELECT "index" FROM {self.dataset_table_name}')
        rows = res.fetchall()
        return [row[0] for row in rows]

    def unique_ids(self) -> List[int]:
        cursor = self.conn.cursor()
        res = cursor.execute(
            f'SELECT "index" FROM {self.dataset_table_name} group by input, instruction, output;'
        )
        rows = res.fetchall()
        return [row[0] for row in rows]

    def clean(self):
        cur = self.conn.cursor()
        cur.execute("DELETE from dataset;")
        self.conn.commit()
