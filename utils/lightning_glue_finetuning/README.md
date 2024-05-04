# Lightning transformer finetuning on GLUE

Langugage classification example with [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/) and 
[🤗 Huggingface Transformers](https://huggingface.co/transformers/).

This repository is based on the colab notebook [04-transformers-text-classification.ipynb](https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/04-transformers-text-classification.ipynb#scrollTo=QSpueK5UPsN7)
utilising the [huggingface](https://huggingface.co/) library for finetuning transformers on the 
[GLUE benchmark](https://gluebenchmark.com/) for text classification.

This script usese a pre-trained transformer model such as [BERT](https://arxiv.org/abs/1810.04805), 
[DistilBERT](https://arxiv.org/abs/1910.01108) or [RoBERTa](https://arxiv.org/abs/1907.11692). For a full list of supported 
models, see the huggingface repository for sequence classification model, [here](https://huggingface.co/transformers/model_doc/auto.html?highlight=automodel#tfautomodelforsequenceclassification).
We finetune the complete model on the GLUE Benchmark. For this the transformer base model is extended with a classifier
and the complete model is adapted for the specific task.

## Setup

```bash
pip install requirements.txt
```

## Use of repository

Running the script `run_glue.py` will finetune a transformer of you choice on a GLUE task of your choice. Run
```bash
python run_glue.py \
--model_name_or_path "String path or name of transformer model to finetune" \
--task_name "String GLUE task name"
--max_epochs "Int Number of epochs" 
```
For example run
```bash
python run_glue.py \
--model_name_or_path "bert-base-cased" \
--task_name "CoLA" \
--max_epochs 3
```
More Pytorch-Lightning specific inputs arguments are possible such as `val_check_interval` for changing the frequency for 
running the validation loop or `fast_dev_run` for a unit test. See the documentation for more.

## Logging

The results are logged in the folder called `\lightning_logs`. This contains checkpoints of the last stored model and a
tensorboard logger of the training and validation progress. To start use
```bash
tensorboard --logdir lightning_logs/
```

