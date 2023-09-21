# GermEval 2023

In this repository, we will shortly share the code of our (Team CPAa) participation in Task 1 (Subtask 1 + 2) of the
GermEval 2023 Shared Task.

## Setup

install pytorch from here: https://pytorch.org/get-started/locally/

install remaining requirements with: `pip install -U -r requirements.txt`

## Fine-tuning

Prepare Llama 2 models in HF (Huggingface) format (either from Huggingface or
from https://github.com/facebookresearch/llama
converted with https://github.com/facebookresearch/llama-recipes/#model-conversion-to-hugging-face)

Prepare data with [parse_data_alpaca_format.ipynb](fine-tuning/scripts/parse_data_alpaca_format.ipynb)

set path to data and path to Llama 2 model in fine-tuning scripts in folder `fine-tuning/scripts/`

set `CUDA_VISIBLE_DEVICES` if you want to limit the used GPUs

set `per_device_train_batch_size` and `gradient_accumulation_steps` so
that `per_device_train_batch_size * gradient_accumulation_steps` is a multiple of 16 and the model fits on your GPU

set `max_steps` to control the length of training (`save_steps` determines when checkpoints are created)

If you want to use the scripts with you own data, you should check the parameters `source_max_len` and `target_max_len`. The [data parsing script](fine-tuning/scripts/parse_data_alpaca_format.ipynb) contains code to determine the maximum length of the source and target sequences in your data. Adapt the values used in the fine-tuning scripts accordingly.

run fine-tuning:

* of 7b cues model: `bash fine-tuning/scripts/finetune_spkatt_7b_cues.sh`
* of 70b cues model: `bash fine-tuning/scripts/finetune_spkatt_70b_cues.sh`
* of 7b roles model: `bash fine-tuning/scripts/finetune_spkatt_7b_roles.sh`
* of 70b roles model: `bash fine-tuning/scripts/finetune_spkatt_70b_roles.sh`
