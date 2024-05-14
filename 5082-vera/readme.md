# ELoRA Source Code

## Structure

### peft

- Modified Hugging Face PEFT library ([original source](https://github.com/huggingface/peft/)).
- The only relevant file for our work is `peft/src/peft/tuners/lora.py`, which includes our method. It also contains additional implementations that were tested but not included in the paper.
- The rest of the library is unchanged.

### glue

- Modified code for running experiments on the GLUE benchmark.
- Adapted from [LoRA source code](https://github.com/microsoft/LoRA/blob/main/examples/NLU/examples/text-classification/run_glue.py).

### instruct

- Code for fine-tuning the llama2 model on the instruction-following dataset.

Both `glue` and `instruct` directories contain scripts for reproducing the results detailed in the paper. For different GLUE tasks, change `task_name`, `classifier_lr`, `learning_rate`, and `num_train_epochs` to corresponding values from the table in Appendix.

## Requirements

- Python 3.10
- Run `pip install -r requirements.txt` to install necessary packages.
- Run `pip install -U ./peft` to install modified peft library.

## Llama2 Instruction-tuning

- Finetuning Llama2 model requires an access to the model weights on HuggingFace. Make sure you have the access before running the code.
- For evaluation, use [Vicuna eval](https://github.com/lm-sys/vicuna-blog-eval) code.

