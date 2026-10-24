# Tiny Alice: A Small Language Model Built from Scratch

A character-level transformer (the same architecture family as GPT) written in PyTorch and trained on *Alice's Adventures in Wonderland*. I built it to understand the full workflow behind a language model: collecting and cleaning text, training, evaluating, and generating.

## What it does

The model learns one task: **given the characters so far, predict the next character.** Repeating that prediction one character at a time lets it generate new text in the style of the book.

## Pipeline

1. **Collect:** plain-text edition of the book from Project Gutenberg (public domain).
2. **Clean (`clean.py`):** remove the Gutenberg license text, title page and table of contents, normalize line endings and blank lines.
3. **Train (`train.py`):** convert characters to numbers, split 90% train / 10% validation, and train a small transformer.
4. **Generate (`generate.py`):** type a starting phrase and the model continues it.

## Model and results

| Item | Value |
|---|---|
| Text size after cleaning | 144,529 characters |
| Vocabulary | 75 unique characters |
| Architecture | 4 transformer layers, 4 attention heads, 128-dimensional embeddings, 128-character context |
| Parameters | 0.83 million |
| Training | 3,000 steps, about 61 minutes on a laptop CPU |
| Train loss | 4.390 → 1.362 |
| Validation loss | 4.391 → 1.521 |

Loss measures how surprised the model is by the real next character (lower is better). Random guessing among 75 characters gives about 4.3.

## Example output

Prompt: `hi there`

> “When you know writh must not itsense,” said the Caterpillied, “and to you would und known a live deepered to here gards!”

## What the output shows

- The model learned the **form** of the book: quotation marks, `said the ...` after speech, paragraph breaks, capitalization, and the main characters' names (Caterpillar, Mock Turtle, Duchess).
- It does **not** learn meaning. Many words are invented, because the model reads one character at a time and has no dictionary.
- It continues text rather than answering it. It treats "hi there" as the start of a passage, not as a message to reply to.

## What I learned

- Train and validation loss moved together early on, then a gap began to open (1.36 vs 1.52), the first sign of the model fitting its training text better than unseen text.
- Data matters most. With one short book, the model can learn style but not language understanding.
- A raw language model is not a chatbot. Assistants need additional training on conversation examples, which is where human-written and human-labeled data come in.
- Temperature changes the output: lower values give more real words but more repetition, higher values give more variety but more gibberish.

## How to run

Requires Python 3.10+ (tested on Windows with PowerShell).

```
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the plain-text (UTF-8) edition of *Alice's Adventures in Wonderland* from [Project Gutenberg](https://www.gutenberg.org/ebooks/11) and save it in this folder as `data.txt`. Then:

```
python clean.py       # creates clean.txt
python train.py       # trains the model (about an hour on a laptop CPU)
python generate.py    # type a starting phrase and see what it writes
```

## Files

| File | Purpose |
|---|---|
| `clean.py` | Cleans the raw text |
| `train.py` | Defines and trains the transformer |
| `generate.py` | Generates text from the trained model |
| `requirements.txt` | Python packages needed |

The book text and trained model files are not included. The text is available from Project Gutenberg, and `train.py` creates the model.

## Notes

I developed the code with AI assistance (Claude) while learning, then ran, tested and analyzed the model myself.

## Possible next steps

- Train on more books, or on conversation-style text, so it can learn to reply instead of only continuing a passage.
- Try a free cloud GPU to train faster and test larger settings.
- Compare different temperature and model-size settings.
