### Installation

```shell
pip install elemelek 
```

### How to use:

Create YAML config file:

```yaml
dataset_jsonl_path: /path/to/your/file.jsonl
db:
  database_insert_batch_size: 1000 
semantic_index:
  embeddings_model_name: sentence-transformers/all-MiniLM-L6-v2
  embeddings_computation_batch_size: 32
  metric: cos
  connectivity: 128
  dtype: 'f32'
  expansion_add: 128
  expansion_search: 256
  n_clusters: 10000
features:
  basic: true
  reranker:
    model_name: cross-encoder/ms-marco-MiniLM-L-2-v2
    batch_size: 16
    strategy: truncate
  language_tool: 
    lang: pl-PL
    n_threads: 4
```

Run 

```python

from elemelek.nest import Elemelek, Egg
from elemelek.settings import RERANKER_RELEVANCE_SCORE
from elemelek.model import SubsetChoiceMethod

# read config file  
egg = Egg.from_yaml("config.yaml")

# create your elemelek 
elemelek = Elemelek(egg)

# list all features 
elemelek.list_features()

# start sampling 
sample = elemelek.as_sample(shuffle=True)

# filter 
sample = sample.filter(lambda x : x.get_feature(RERANKER_RELEVANCE_SCORE).value > 0.9)
sample = sample.sample_diverse(
    k=100000, 
    method = SubsetChoiceMethod.TARGET_MEDIAN_DISTANCE, 
    target_median=0.1
)
sample = sample.stratify("source_name", 20000)

df = sample.to_pandas()
```
