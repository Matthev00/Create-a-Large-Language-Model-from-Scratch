# Project files
chat_bot.py - script for using chat bot

train_py - script for training chat bot

data_prep.py -script for preparing data from OpenWebText

model_builder.py - basicly classes to recreate GPT architecture and func to create model

utils.py - useful functions

gpt-v1.ipynb - proces of my learnig learnig LLM's

# Model architecture
![Alt text](Model_Architecture.png)

# To do
## Data
+ fine tune using datasets from hugging face
+ another dataset is needed
## Training
+ reate new test and train step for training loop
+ train for longer
+ train on basic data for next 4000 iters
+ finetune to the medical version

# Isssues
finetuning dameges results on basic model
fatal results of finetuning
model should take 1 question 1 answer and then in loss function it shoudl compera answer to answer not all tehst

# Resources
## Create-a-Large-Language-Model-from-Scratch
Based on https://www.youtube.com/watch?v=UU1WVnMk4E8&amp;list=WL&amp;index=10&amp;ab_channel=freeCodeCamp.org

https://huggingface.co/datasets/katielink/healthsearchqa/viewer/all_data
"E:\projekty python\Create-a-Large-Language-Model-from-Scratch\data\finetuning_med\41586_2023_6291_MOESM6_ESM.xlsx"
E:\projekty python\Create-a-Large-Language-Model-from-Scratch\data\finetuning_med\raw

time 4:02:09

## Research Papers
Attention is All You Need - https://arxiv.org/pdf/1706.03762.pdf

A Survey of LLMs - https://arxiv.org/pdf/2303.18223.pdf

QLoRA: Efficient Finetuning of Quantized LLMs - https://arxiv.org/pdf/2305.14314.pdf

https://towardsdatascience.com/fine-tuning-large-language-models-llms-23473d763b91
https://www.ml6.eu/blogpost/to-fine-tune-or-not-to-fine-tune-that-is-the-question - read
https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset
https://bdtechtalks.com/2023/01/16/what-is-rlhf/

## Data resources
OpenWebText Download - https://skylion007.github.io/OpenWebTextCorpus/
