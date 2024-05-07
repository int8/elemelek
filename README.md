### Installation

```shell
pip install elemelek 
```


### What does elemelek do ? 

Elemelek designed to sample subsets of instructions gathered from various sources (and so with various quality/diversity) 
for LLM fine-tuning tasks. Under the hood elemelek does the following: 

- creates sqlite database to keep instructions / features at
- computes embeddings of instructions 
- index these embeddings in HNSW index via `usearch`
- cluster embeddings  
- compute features of each instruction in dataset (basic text statistics + rerank score)


### How to use it:

First you need to "build" your dataset. 

Create YAML config file:

```yaml
dataset_jsonl_path: /path/to/your/file.jsonl
db:
  database_insert_batch_size: 1000 # chunk size dataset will be written to db with 
semantic_index:
  embeddings_model_name: sentence-transformers/all-MiniLM-L6-v2 # sentence-transformer model used to compute embeddings of instructions 
  embeddings_computation_batch_size: 32 # batch-size used for embeddings computation 
  metric: cos # metric used for HNSW index 
  connectivity: 128 # HNSW connectivity parameter  
  dtype: 'f32' # index dtype (`f16` `i8` could be used for perfomance reasons)  
  expansion_add: 128  # expansion factor used for index construction when adding vectors
  expansion_search: 256 # expansion factor used for index construction during search operations.
  n_clusters: 10000 # number of clusters to compute once index is created 
features:
  basic: true # whether or not to compute basic features 
  reranker:
    model_name: cross-encoder/ms-marco-MiniLM-L-2-v2 # reranker to score relevance of (instruction, input) => output pairs 
    batch_size: 16 # batch size used to reranking computation 
    strategy: truncate # strategy of how to treat long text (currently truncate only) 
  language_tool: # set it to false if you don't need this one  [a bit experimental + it takes a while] 
    lang: pl-PL # language of your instructions
    n_threads: 4 # number of threads that will check your texts with language_tool 
```

Run 

```python

from elemelek.nest import Elemelek, Egg
# read config file  
egg = Egg.from_yaml("config.yaml")
# create your elemelek - this will take a bit 
elemelek = Elemelek(egg)
```

Once your dataset is built you can start sampling 

```python
from elemelek.settings import RERANKER_RELEVANCE_SCORE
from elemelek.model import SubsetChoiceMethod
# start sampling 
sample = elemelek.start_sampling(shuffle=True)

# filter 
sample = sample.filter(lambda x : x.get_feature(RERANKER_RELEVANCE_SCORE).value > 0.9)

# sample 10k points with distance between them to be targeted at around 0.1
sample = sample.sample_diverse(
    k=100000, 
    method = SubsetChoiceMethod.TARGET_MEDIAN_DISTANCE, # for each cluster 
    target_median=0.1
)

# get 20k instructions following uniform distribution of categorical feature "source_name"  
sample = sample.stratify("source_name", 20000)

# get DF and play with it 
df = sample.to_pandas()

# dump your data to JSONL (and hopefully train your great fine-tuned LLM)
sample.to_jsonl("my-awasome-sample.jsonl")
```

Additionally, you can 
```python
# search through your instructions semtantically  
matched_instructions = elemelek.search("How much wood would the woodchuck chuck?",  k = 10)

# examine clustering requested in your config 
clustering = elemelek.clustering
centroid_instruction_id = clustering[0].centroid_id
example_centroid_instruction = elemelek.db[centroid_instruction_id] # access your instruction like this 

# list all precomputed feature names  
feature_names = elemelek.feature_names


```

Once you are done you can resume your work later 

```python
from elemelek.nest import Elemelek
datasets = Elemelek.list_datasets()
# >> {'7ff7a3107f44d545c9ac6703c3893e0b': Egg(...)}
elemelek = Elemelek.from_dataset_id('7ff7a3107f44d545c9ac6703c3893e0b')
```

have fun