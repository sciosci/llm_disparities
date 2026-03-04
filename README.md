# The Disparate Beliefs of Large Language Models About Science and Scientists

This repository contains the data, code, and analyses associated with the paper:

**The Disparate Beliefs of Large Language Models About Science and Scientists**  
Almene D. M. Meguimtsop, Christopher E. Ojukwu, Pawin Taechoyotin, Carolina Chávez-Ruelas, Robin Burke, Aaron Clauset, and Daniel E. Acuna

---

## Overview

Large Language Models (LLMs) are increasingly used in scientific workflows, including literature search, hypothesis generation, peer review assistance, and research synthesis. However, little is known about how these systems internally represent scientific communities or how they generate beliefs about scientists.

This project investigates whether **systematic disparities emerge in LLM representations and generated outputs related to science and scientists.**

We analyze two complementary dimensions of LLM behavior:

1. **Embedding Representations using the Word Embeddings Association Test (WEAT) framework**. The models audited include BERT, SciBERT, and OpenAI’s text-embedding-3-large.
2. **Generated Outputs via Role-play Prompting**. We evaluated Claude 3.5 Sonnet (version 1 and version 2 released on June 20, 2024 and October 22, 2024 respectively) and Claude 3.7 Sonnet (version 1, released on February 24, 2025) from Anthropic, Llama 3-70b instruct (version 1, released on April 18, 2024) from Meta, and OpenAI’s GPT- 4o

These analyses allow us to study how LLMs encode and express beliefs about science, scientists, and scientific institutions.

---

## Research Questions

This work investigates the following questions:

1. Do LLM embeddings encode disparities about scientists or scientific research and academia?
2. Do LLM-generated outputs reproduce systematic differences in beliefs about science?
3. How do these disparities vary across geographic or institutional contexts?

## Reproducibility

To reproduce the experiments:

#### 1- Clone the repository


git clone https://github.com/sciosci/llm_disparities.git

cd llm_disparities

#### 2- Install dependencies


pip install -r requirements.txt

#### 3- Run analysis

Execute the scripts or notebooks within each analysis directory.

Results will be saved in the corresponding results folders.

---

## Dependencies

The project uses common scientific Python libraries:

- Python ≥ 3.9
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- transformers
- torch
- openai
- anthropic
- boto3


---

## Results

Results are stored in CSV format and include aggregated metrics derived from LLM responses.

These results are used to produce figures and statistical analyses presented in the paper.

---

## Ethical Considerations

Large language models can reproduce disparities present in training data. This research aims to better understand how such disparities manifest in scientific contexts.

The goal of this work is **measurement and analysis**, not reinforcement of stereotypes or harmful narratives.