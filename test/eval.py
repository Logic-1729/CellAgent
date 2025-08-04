"""
This examples trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) for the STSbenchmark from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure the similarity.

Usage:
python training_stsbenchmark.py

OR
python training_stsbenchmark.py pretrained_transformer_model_name
"""

import logging
import sys
import traceback
from datetime import datetime
from datasets import load_dataset

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# You can specify any Hugging Face pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else "/data_afs_c_mtc/fengdahu/gongwangbo/output/v1-20250731-130317/checkpoint-3500"

# 1. Here we define our SentenceTransformer model. If not already a Sentence Transformer model, it will automatically
# create one with "mean" pooling.
model = SentenceTransformer(model_name)

# 2. Load the STSB dataset: https://huggingface.co/datasets/sentence-transformers/stsb
eval_dataset = load_dataset("csv",data_files = "sts_mybench_data.csv")["train"]
test_dataset = load_dataset("csv",data_files = "sts_mybench_data.csv")["train"]


# 7. Evaluate the model performance on the STS Benchmark test dataset
test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=test_dataset["query"],
    sentences2=test_dataset["response"],
    scores=test_dataset["label"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-test",
)
test_evaluator(model)
