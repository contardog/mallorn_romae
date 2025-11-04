#  Mallorn with RoMAE


The mallorn_romae is a pip-installable package :
> pip install -e .

Which provides some code / functions to deploy RoMAE  (require installation from https://github.com/Chromeilion/RoMAE ; paper: https://arxiv.org/abs/2505.20535 ) for the Mallorn challenge, following the org/architecture provided in RoMAE-experiments repo : https://github.com/Chromeilion/RoMAE-Experiments 

The notebook gives some code to transform Mallorn data into parquet files handled with polars library.

The code should roughly work?

You can call for pre-training (self-supervised only, no use of labels):

>python -m romae_mallorn pretrain --train_parquet /path/to/MALLORN/train_mallorn_data.parq --test_parquet /path/to/MALLORN/test_mallorn_data.parq --model_name mallorn_test1_supertiny --batch_size=128 --epochs=100 --model_size=brutally-mini

For fine-tuning classification (setup right now as binary):

>python -m romae_mallorn finetune --train_parquet /path/to/MALLORN/train_mallorn_data.parq --test_parquet /path/to/MALLORN/test_mallorn_data.parq --model_name mallorn_test_finetune1_supertiny --batch_size=16 --epochs=100 --pretrained_model=/path/to/mallorn_test1_supertiny_checkpoint_/checkpoint-2000/

For evaluation:
>python -m romae_mallorn evaluate --test_parquet /path/to/MALLORN/test_mallorn_data.parq --eval_checkpoint=/path/to/mallorn_test_finetune1_supertiny_checkpoint-finetune_/checkpoint-2400/



TODO add some info and link e.g. for model sizes and stuff

