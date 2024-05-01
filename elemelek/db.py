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
from elemelek.utils import count_jsonl_objects
from collections.abc import Sequence


class InMemoryFeatures(SelfLogging):
    def __init__(self, features_json_path: str):
        self.features_json_path = features_json_path
        if os.path.exists(self.features_json_path):
            with open(self.features_json_path, "r") as f:
                self._data = json.load(f)
                self._data = {int(k): v for k, v in self._data.items()}

        else:
            self._data: Dict[int, Dict[str, InstructionFeature]] = dict()

    def add(self, features: List[InstructionFeature], save: bool = True):
        for feature in features:
            self._data[feature.instruction_id] = self._data.get(
                feature.instruction_id, dict()
            ) | {feature.name: feature.value}

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

    def load_jsonl(self, jsonl_path: str, chunksize: int = 25000):
        try:
            reader = pd.read_json(jsonl_path, lines=True, chunksize=chunksize)

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
                chunk[list(MANDATORY_FIELDS)].to_sql(
                    self.dataset_table_name, self.conn, if_exists="append"
                )
                other_columns = set(chunk.columns) - MANDATORY_FIELDS
                if len(other_columns) > 0:
                    for column in other_columns:
                        other_features = [
                            InstructionFeature(
                                name=column, instruction_id=id_, value=value
                            )
                            for id_, value in chunk[column].items()
                        ]
                        self.features.add(other_features)

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
        return self.__yield_rows()

    def __getitem__(self, idx: int | slice) -> Instruction | List[Instruction]:
        cur = self.conn.cursor()

        if isinstance(idx, slice):
            indices = str(tuple(range(0, len(self)))[idx])
            res = cur.execute(
                f'SELECT "index", instruction, input, output FROM {self.dataset_table_name} WHERE "index" in {indices}'
            )
            instructions = []
            rows = res.fetchall()
            for row in rows:
                id_, instruction, input_, output = row
                instructions.append(
                    Instruction(
                        id=id_,
                        instruction=instruction,
                        input=input_,
                        output=output,
                        features=self.features.get_features(instruction_id=id_),
                    )
                )
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

    def insert_features(self, features: List[InstructionFeature]):
        self.features.add(features)

    def __yield_rows(self) -> Instruction:
        cursor = self.conn.cursor()
        cursor.execute(
            f'SELECT "index", instruction, input, output FROM {self.dataset_table_name}'
        )
        row = cursor.fetchone()
        while row:
            row = cursor.fetchone()
            if row:
                id_, instruction, input_, output = row
                yield Instruction(
                    id=id_,
                    instruction=instruction,
                    input=input_,
                    output=output,
                    features=self.features.get_features(instruction_id=id_),
                )
        cursor.close()

    def clean(self):
        cur = self.conn.cursor()
        cur.execute("DELETE from dataset;")
        self.conn.commit()
