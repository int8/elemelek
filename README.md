### Installation

```shell
pip install elemelek 
```


### What does elemelek do ? 

Elemelek is designed to sample subsets of instructions gathered/generated from various sources (and so with various quality/diversity) 
for LLM fine-tuning tasks. Under the hood elemelek does the following: 

- creates sqlite database to keep instructions / features in
- computes embeddings of instructions 
- indexes the embeddings in HNSW index via `usearch`
- clusters the embeddings 
- compute features of each instruction in dataset (basic text statistics + rerank score)

Once created it provides simple interface to sample filtered data 

### How to use it:

First you need to "build" your dataset. 

Create YAML config file:

```yaml
dataset_jsonl_path: /path/to/your/file.jsonl
db:
  database_insert_batch_size: 1000 # chunk size dataset will be written to db with 
  remove_duplicates: true # do not keep duplicated entries in database 
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
# create your elemelek - this will take a moment, strong GPU is required for embeddings and rerank relevance scores computation 
elemelek = Elemelek(egg)
```

Once your dataset is built you can start sampling

```python
from elemelek.features import RERANKER_RELEVANCE_SCORE, IS_QUESTION
from elemelek.model import SubsetChoiceMethod

# start sampling 
sample = elemelek.start_sampling(shuffle=True)

# filter questions with relevance score > 0.9 
substantive_questions  = sample.filter(
    lambda x: ( 
        x.get_feature(IS_QUESTION).value == True and 
        x.get_feature(RERANKER_RELEVANCE_SCORE).value > 0.9
    )
)

# get subset of 25k diverse substantive questions  
diverse_questions = substantive_questions.sample_diverse(
    k=25000,
    method=SubsetChoiceMethod.VARIABILITY_FACTOR,  
    within_cluster_diversity_factor=1.0
    # within_cluster_diversity_factor=0.0 => the least diverse subset
    # within_cluster_diversity_factor=1.0 => the most diverse subset 
)

# get non-questions 
non_questions = sample.filter(
    lambda x: x.get_feature(IS_QUESTION).value == False, 
)

# get 25k diverse non questions 
diverse_non_questions = non_questions.sample_diverse(
    k=25000,
    method=SubsetChoiceMethod.VARIABILITY_FACTOR,  
    within_cluster_diversity_factor=1.0
)

# compose final sample 
final_sample = diverse_non_questions + diverse_questions

# get DF and play with it 
df = final_sample.to_pandas()

# dump your data to JSONL (and hopefully train your great fine-tuned LLM)
# features will be included to output jsonl + you will find __instruction_text field 
# representing formatted instruction (using apply_chat_template from tokenizer of your choice) 
final_sample.to_jsonl(
    "my-awasome-sample.jsonl", 
    include_features=True, 
    include_instruction_using_chat_template_from="mistralai/Mistral-7B-Instruct-v0.2"
)
```

Your awesome-sample.jsonl entries will look like this: 

```json
{
  "id": 476137,
  "instruction": "Jakie są produkty z mleka?",
  "input": "",
  "output": "Do nabiału należą również produkty mleczne...",
  "feature_source_name": "almost_like_an_alpaca",
  "feature_category": "ALMOST_LIKE_AN_ALPACA",
  "feature_median_word_length": 6,
  "feature_quantile_0.9_word_length": 9.6,
  "feature_quantile_0.1_word_length": 1.8,
  "feature_total_length": 292,
  "feature_is_question": true,
  "feature_has_input": true,
  "feature_numeric_chars_ratio": 0,
  "feature_non_alpha_numeric_chars_ratio": 0.18835616438356165,
  "feature_reranker-relevance-score": 0.9645681381225586,
  "__instruction_text": "<s>[INST] Jakie są produkty z mleka? \n  [/INST]Do nabiału należą również produkty mleczne...</s> "
}


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
