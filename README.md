# Vritya-Tiny-163M

A 163M parameter GPT-style language model built from scratch, pretrained on TinyStories and finetuned for two tasks:

- **Story Generation** -- Instruction-finetuned on TinyStoriesInstruct to generate children's stories from prompts
- **Spam Classification** -- Finetuned on the SMS Spam Collection dataset to classify messages as ham or spam

Model weights are hosted on [Hugging Face](https://huggingface.co/curious-techie/Vritya-Tiny-163M) and downloaded automatically on first run.

## Project Structure

```
pretraining.py                 # Pretrain the base GPT model on TinyStories
instruction-finetuning.py      # Finetune for story generation (TinyStoriesInstruct)
classification-finetuning.py   # Finetune for spam classification (SMS Spam Collection)
generate.py                    # Standalone script for story generation inference
classify.py                    # Standalone script for spam classification inference
app.py                         # Flask web app with both models
requirements.txt               # Python dependencies
```

## Setup

```bash
git clone https://github.com/VrityaCodeRishi/Vritya-Tiny-163M-1.git
cd Vritya-Tiny-163M-1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Web App

```bash
python3 app.py
```

Open `http://localhost:5001` in your browser. The app has two modes:

- **Spam Classifier** -- Type any text message to classify it as ham or spam
- **Instruction Following** -- Type a story prompt and adjust temperature, repetition penalty, and max tokens

Model weights are downloaded from Hugging Face automatically on first launch.

## Running the CLI Scripts

**Story generation:**

```bash
python3 generate.py
```

Runs a few demo prompts, then enters interactive mode where you can type instructions.

**Spam classification:**

```bash
python3 classify.py
```

Classifies demo messages, then enters interactive mode for custom messages.

## Training from Scratch

**1. Pretrain the base model:**

```bash
python3 pretraining.py
```

Pretrains on the TinyStories dataset. Saves `best_model.pth`.

**2. Instruction finetune for story generation:**

```bash
python3 instruction-finetuning.py
```

Finetunes the pretrained model on TinyStoriesInstruct. Saves `instruction_finetuned.pth`.

**3. Finetune for spam classification:**

```bash
python3 classification-finetuning.py
```

Finetunes the pretrained model on the SMS Spam Collection. Saves `finetuned_for_classification.pth`.

## Model Weights

All weights are hosted at [huggingface.co/curious-techie/Vritya-Tiny-163M](https://huggingface.co/curious-techie/Vritya-Tiny-163M):

| File | Description | Size |
|------|-------------|------|
| `best_model.pth` | Pretrained base model | 622 MB |
| `instruction_finetuned.pth` | Story generation finetuned | 622 MB |
| `finetuned_for_classification.pth` | Spam classifier finetuned | 475 MB |

## Hardware

The code automatically selects the best available device: CUDA > MPS > CPU.
