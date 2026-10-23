# Neural Machine Translation

A from-scratch **English-to-Hindi Neural Machine Translation (NMT)** system developed using the IIT Bombay English-Hindi Parallel Corpus.

The objective of this project is to build a complete neural translation pipeline that accepts an English sentence and generates its Hindi translation.

The final system uses a **Refined Transformer** architecture trained from scratch. Earlier BiLSTM and baseline Transformer implementations were used during model development and comparative experimentation, while the Refined Transformer was selected as the final translation model.

---

## Project Objective

The primary objective is:

> **To develop an English-to-Hindi Neural Machine Translation system from scratch and obtain a reliable final model for generating Hindi translations from English text.**

The project covers the complete NMT pipeline:

```text
English-Hindi Parallel Corpus
            ↓
      Data Cleaning
            ↓
 SentencePiece Tokenization
            ↓
   Sequence Preparation
            ↓
      Model Training
            ↓
   Model Development and
       Comparison
            ↓
 Final Refined Transformer
            ↓
      Beam Search
            ↓
     Hindi Translation
```

No pretrained language models, pretrained embeddings, pretrained encoders, or pretrained translation models are used.

---

# Final Model

The final translation system is a **Refined Transformer encoder-decoder model**.

Its main configuration is:

```text
Encoder layers        : 6
Decoder layers        : 6
Model dimension       : 384
Attention heads       : 6
Feed-forward dimension: 1536
Dropout               : 0.1
Activation            : GELU
Normalization         : Pre-LayerNorm
Vocabulary            : 16,000 English + 16,000 Hindi
Tokenizer             : SentencePiece Unigram
```

The final model additionally uses:

- sinusoidal positional encoding
- target embedding-output weight tying
- label smoothing during training
- development-set chrF++ based model selection
- top-5 checkpoint averaging
- beam-search decoding

The final averaged model contains approximately:

```text
37.15 million trainable parameters
```

---

## Final Translation Performance

The final model is evaluated on the complete **2,507-sentence test set**.

### Final Refined Transformer

| Metric | Result |
|---|---:|
| Cross-Entropy loss | 2.3510 |
| Perplexity | 10.4965 |
| Token Accuracy | 54.80% |
| BLEU | **17.6765** |
| chrF++ | **43.4346** |
| TER | **68.1295** |

Final decoding configuration:

```text
Beam size      : 2
Length penalty : 1.0
```

Beam-search parameters were selected using the development set before final test evaluation.

---

# Example Usage

The trained model can translate English text directly from the command line.

```bash
python transformer/translate.py \
    --checkpoint averaged_5_dev_chrf.pt \
    --method beam \
    --text "This is a test sentence."
```

The model generates the corresponding Hindi translation.

---

# Web Translator

> **The web interface is used for presentation as hindi text was not correctly redered on terminal.**


A simple browser interface is also provided.

Run:

```bash
python transformer/web_translate.py \
    --checkpoint averaged_5_dev_chrf.pt
```

Then open:

```text
http://127.0.0.1:7860
```

The interface provides:

```text
English Input
      ↓
Final NMT Model
      ↓
Hindi Translation
```

Both greedy and beam decoding can be selected from the interface.

---

# Dataset

uses **IIT Bombay English-Hindi Parallel Corpus**.

Dataset statistics used in the project:

```text
Raw training pairs             : 1,659,083
After cleaning                 : 1,391,221
Final training pairs           : 1,390,186
Development pairs              : 520
Test pairs                     : 2,507
```

Training sequences exceeding the configured maximum SentencePiece-token length are removed.

Development and test sets are kept separate from training throughout the pipeline.

---

# Tokenization

English and Hindi use independently trained **SentencePiece Unigram tokenizers**.

Configuration:

```text
Vocabulary size      : 16,000 per language
Model type           : Unigram
Training sentences   : 500,000
Character coverage   : 0.9999
```

Special token IDs:

```text
UNK = 0
BOS = 1
EOS = 2
PAD = 3
```

The sequence representation used by the translation model is:

```text
Source:
[x1, x2, ..., xn, EOS]

Decoder Input:
[BOS, y1, y2, ..., ym]

Target:
[y1, y2, ..., ym, EOS]
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/rishabh-jain89/Neural-Machine-Translation.git
cd Neural-Machine-Translation
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# Download the Trained Model

Large trained checkpoints are distributed through the repository's **GitHub Releases** rather than directly through Git.

Download the release assets containing:

```text
refined_transformer_seed19.tar.gz
sentencepiece_tokenizers.tar.gz
```

The final translation model should be placed under:

```text
output/checkpoints/refined_transformer_seed19/
```

The required tokenizer models should be placed under:

```text
output/models/
```

Expected final files include:

```text
output/checkpoints/refined_transformer_seed19/
└── averaged_5_dev_chrf.pt

output/models/
├── en_unigram_500000_16000_0.9999.model
└── hi_unigram_500000_16000_0.9999.model
```

---

# Run the Final Model

For beam-search translation:

```bash
python transformer/translate.py \
    --checkpoint averaged_5_dev_chrf.pt \
    --method beam \
    --text "Machine translation is an important application of natural language processing."
```

For greedy decoding:

```bash
python transformer/translate.py \
    --checkpoint averaged_5_dev_chrf.pt \
    --method greedy \
    --text "Machine translation is an important application of natural language processing."
```

Beam search is used as the final decoding strategy.

---

# Final Model Evaluation

To reproduce development-set beam selection:

```bash
python transformer/evaluate_transformer.py \
    --mode tune-beam \
    --run-name refined_transformer \
    --checkpoint averaged_5_dev_chrf.pt
```

Then run the final test evaluation:

```bash
python transformer/evaluate_transformer.py \
    --mode test \
    --run-name refined_transformer \
    --checkpoint averaged_5_dev_chrf.pt
```

The generated results are stored under:

```text
output/evaluation/refined_transformer/
```

including:

```text
beam_tuning_dev.json
test_metrics.json
test_predictions.csv
```

---

# Training the Final Model

The final Transformer can be trained using:

```bash
python transformer/train_transformer.py \
    --config transformer/configs/refined.json
```

The training configuration defines the architecture, optimizer, learning-rate schedule, early stopping, model-selection criterion, and checkpoint behavior.

After training, the strongest development checkpoints can be averaged using:

```bash
python transformer/average_checkpoints.py
```

The final model used in this project is the average of the five checkpoints selected by development-set chrF++.

Selected epochs:

```text
12
13
15
16
17
```

---

# Complete Pipeline

The project contains scripts for reproducing the complete workflow.

### Data preprocessing

```bash
python preprocess/clean_train.py
python preprocess/clean_eval.py
```

### Dataset auditing

```bash
python scripts/audit_raw_data.py
python scripts/audit_char_coverage.py
python scripts/sequence_representation.py
```

### Tokenizer training

```bash
python tokenizer/tokenizer.py
```

### Tokenizer audit

```bash
python tokenizer/tokenizer_audit.py
```

### Final model training

```bash
python transformer/train_transformer.py \
    --config transformer/configs/refined.json
```

### Final model evaluation

```bash
python transformer/evaluate_transformer.py \
    --mode tune-beam \
    --run-name refined_transformer \
    --checkpoint averaged_5_dev_chrf.pt

python transformer/evaluate_transformer.py \
    --mode test \
    --run-name refined_transformer \
    --checkpoint averaged_5_dev_chrf.pt
```

---

# Development Experiments

During development, additional architectures were also implemented before the final model.

The progression was:

```text
BiLSTM + Bahdanau Attention
            ↓
   Baseline Transformer
            ↓
    Refined Transformer
            ↓
       Final NMT Model
```

---

# Evaluation Metrics

The final system is evaluated using standard machine translation metrics.

### BLEU

Measures n-gram overlap between generated and reference translations.

```text
Higher is better.
```

### chrF++

Measures character-level and word-level similarity between the generated translation and the reference.

```text
Higher is better.
```

### TER

Translation Edit Rate measures the amount of editing required to transform the generated translation into the reference.

```text
