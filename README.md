# NLP Course

This is the course repo for Natural Language Processing.

## History

### 24.08.2026

Added:

- `index.html` — language selection page (EN / KZ / RU)
- `page/css/styles.css` — shared stylesheet for all pages
- `page/js/course.js` — course contents: the lecture list that builds the sidebar and the roadmap
- `page/js/main.js` — shared JavaScript: sidebar navigation, copy buttons, Regex playground, quizzes, progress bar
- `language/en/index.html`, `language/kk/index.html`, `language/ru/index.html` — course roadmap
- `language/en/00-basics-nlp.html`, `language/kk/00-basics-nlp.html`, `language/ru/00-basics-nlp.html` — Lecture 0, Basics of NLP
- `language/en/01-regular-expressions.html`, `language/kk/01-regular-expressions.html`, `language/ru/01-regular-expressions.html` — Lecture 1, Regular Expressions

Changed:

- `language/kk/*.html` — added a machine-translation notice (red banner at the top of each Kazakh page and a compact note in the sidebar)

### 25.08.2026

Added:

- `language/en/02-text-processing.html`, `language/kk/02-text-processing.html`, `language/ru/02-text-processing.html` — Lecture 2, Text Processing (Tokenization, Lemmatization, Bag of Words, TF-IDF): theory first, then practice

Changed:

- `page/js/main.js` — new interactive text-processing lab (normalize → tokenize → stopwords → Bag of Words matrix)
- `page/css/styles.css` — styles for the text-processing lab
- `page/js/course.js` — Lecture 2 marked as published
- `index.html` — link to Lecture 2
- `language/*/02-text-processing.html` — added a runnable Python snippet after each theory explanation (13 per language), including how to handle what each step destroys: keeping hashtags and emphasis, repairing mixed Cyrillic/Latin scripts, protecting times and phone numbers before tokenizing, keeping hyphenated names, splitting sentences without breaking on initials, a mini-BPE, a working suffix stripper, stopword removal that preserves negation, cosine similarity, TF-IDF ranking, min_df pruning

### 26.08.2026

Added:

- `code_files/lesson3.py` — the Lecture 3 script: KazSAnDRA polarity classification in six steps (load, explore, balance, train, test, inspect), Naive Bayes written from scratch, optional scikit-learn comparison
- `language/en/03-classical-ml.html`, `language/kk/03-classical-ml.html`, `language/ru/03-classical-ml.html` — Lecture 3, Classical Machine Learning for Text Data

Changed:

- `page/css/styles.css` — styles for inline SVG figures and the algorithm comparison table
- `page/js/course.js` — Lecture 3 marked as published
- `index.html` — link to Lecture 3

### 30.08.2026

Changed:

- `code_files/lesson3.py` — rewritten around the real KazSAnDRA dataset (`issai/kazsandra`, polarity split): EDA first, then balanced train and test sets, then the models; simpler and linear
- `language/*/03-classical-ml.html` — every number now comes from KazSAnDRA instead of a hand-written corpus: 0.789 accuracy on a balanced 2 000-review test set, the effect of skipping balancing, the learning curve, cross-validation, the smoothing sweep and error analysis
- `.gitignore` — `.env` and `code_files/data/` (the downloaded, gated CSVs)

Changed later the same day:

- `code_files/lesson3.py` — everything now runs on scikit-learn (TF-IDF + Pipeline, MultinomialNB / LogisticRegression / LinearSVC, `classification_report`, `cross_val_score`, `ConfusionMatrixDisplay`); the HF token is read with `python-dotenv`
- `code_files/requirements.txt` — added
- `language/*/03-classical-ml.html` — all numbers regenerated from the scikit-learn run: Naive Bayes 0.784, Logistic Regression 0.780, Linear SVM 0.762 on a balanced 2 000-review test set
- `code_files/lesson3.py` — the dataset is now read with `datasets.load_dataset(...).to_pandas()` instead of manual downloads, and two more models were added: `DecisionTreeClassifier` and `RandomForestClassifier`
- `code_files/requirements.txt` — `datasets` added
- `language/*/03-classical-ml.html` — the comparison now covers five models: Naive Bayes 0.784, Logistic Regression 0.780, Random Forest 0.775, Linear SVM 0.762, Decision Tree 0.750

### 01.09.2026

Added:

- `code_files/lesson4.py` — the Lecture 4 script: trains word2vec and fastText on all 134 368 KazSAnDRA reviews with gensim, inspects what the vectors learned, then swaps `TfidfVectorizer` for a `MeanEmbedding` transformer inside the Lecture 3 pipeline; the trained models are cached in `code_files/models/`
- `language/en/04-word-embeddings.html`, `language/kk/04-word-embeddings.html`, `language/ru/04-word-embeddings.html` — Lecture 4, Word Embeddings: the distributional hypothesis, word2vec (skip-gram, CBOW, negative sampling), gensim, fastText and character n-grams, embeddings for Kazakh, document vectors, and the measured comparison against TF-IDF

Changed:

- `code_files/requirements.txt` — `gensim` added
- `page/js/course.js` — Lecture 4 marked as published, with its Kazakh and Russian titles
- `index.html` — link to Lecture 4

### 02.09.2026

Changed:

- `language/*/04-word-embeddings.html` — a new opening section, "What an embedding is", placed before everything else: a word as a list of numbers, a hand-written two-dimensional table you can read, cosine similarity as an angle, the real 100-number vectors (and the demonstration that no single dimension means anything), and a plain-language walkthrough of where the numbers come from — random start, sliding window, nudge, repeat. The old one-line definition and the vector-shape demo were folded into it, and the freed-up slot now compares the TF-IDF matrix (75 352 000 cells, 0.09% non-zero) with the embedding table (2 707 600 numbers, all used)

### 02.09.2026 (later)

Changed:

- `code_files/lesson4.py` — rewritten as a standalone lesson on **training** embeddings. It no longer imports anything from `lesson3.py`: it loads KazSAnDRA itself, tokenizes it, trains word2vec and then fastText, saves both to `models/word2vec.model` and `models/fasttext.model`, and only in the last step loads them back to reuse as features for the Lecture 3 classifier. New `--retrain` and `--min-count` flags
- `language/*/04-word-embeddings.html` — restructured around that flow. The "Embeddings for Kazakh" section (pretrained `cc.kk.300` and friends) was removed; the lecture now works only with word2vec and fastText that we train ourselves. Two new sections, "Preparing the corpus" and "Training word2vec with gensim", each parameter explained; fastText is now trained and saved on the page too; a new "Reusing the models in the Lecture 3 classifier" section holds one complete, self-contained script, and "Did it help?" became pure analysis of its output
- `language/*/04-word-embeddings.html` — every code example is now self-contained: no `from lesson4 import …` or `from lesson3 import …` anywhere. Each snippet either loads the data itself or loads a saved model with `Word2Vec.load("models/word2vec.model")`, which is also the point being taught
- `language/*/04-word-embeddings.html` — corrected the idf-weighting figure: measured again, it is 0.787 for word2vec and 0.789 for fastText, not the 0.765 previously stated. Knowledge-check question 5 and practice task 8 were replaced, since both were about pretrained Kazakh vectors

### 07.09.2026

Added:

- `code_files/lesson5.py` — the Lecture 5 script: recurrent networks in PyTorch, run on both KazSAnDRA tasks (`polarity_classification` with two classes and `score_classification` with five). Vocabulary, padding and `pack_padded_sequence`, an `Embedding → RNN/LSTM → Linear` model, a hand-written training loop, and `--pretrained` to start the embedding layer from the Lecture 4 word2vec. Pinned to CPU with a fixed seed and a fresh DataLoader generator per model, so every run reproduces
- `language/en/05-rnn-lstm.html`, `language/kk/05-rnn-lstm.html`, `language/ru/05-rnn-lstm.html` — Lecture 5, Text Classification (with recurrent networks). What word order costs a bag of words (measured), the hidden state, the vanishing gradient, the LSTM cell state and its three gates, then the full PyTorch build and the results: RNN 0.755, LSTM 0.772, LSTM from the Lecture 4 vectors 0.787, and 0.803 on 40 000 reviews — the first neural result to beat Lectures 3 and 4. The five-class task lands at 0.374, with the confusion matrix and the within-one-star reading that accuracy alone cannot give

Changed:

- `code_files/requirements.txt` — `torch` added
- `page/js/course.js` — Lecture 5 published as "Text Classification" / "Мәтінді жіктеу" / "Классификация текста"
- `index.html` — link to Lecture 5

### 07.09.2026 (later)

Changed:

- `language/*/05-rnn-lstm.html` — the theory sections rewritten around visuals, with six new runnable demonstrations and four new diagrams:
  - **How the hidden state is passed** — a diagram of one RNN step annotated with `hₜ = tanh(W·xₜ + U·hₜ₋₁ + b)`, then the recurrence run by hand in a loop and checked against `nn.RNN`, then a coloured grid of the four state numbers redrawn after every word
  - **The vanishing gradient, measured** — one signal sent back from the final state of a 40-word sequence, printing how much reaches each position: the RNN keeps ×0.48 per step (5·10⁻¹³ at the far end), an untrained LSTM ×0.60, and the same LSTM with the forget gate held open ×1.00 — the honest point being that the architecture alone fixes nothing, it only adds a knob the training can turn. Plotted on a log axis from the measured numbers
  - **The LSTM cell** — a full diagram of the cell-state path through one multiplication and one addition, the four gate equations term by term, and the gates printed step by step from a hand-written implementation checked against `nn.LSTM`
  - **Batching and padding** — why a rectangle is needed at all, the three decisions in `collate`, a diagram of the padded batch turning into `data` + `batch_sizes`, and a measurement of what breaks without packing (padding changes the final state by 0.583; `pack_padded_sequence` reproduces the honest answer exactly)
  - **The network end to end** — a layer-by-layer diagram with the tensor shape between each pair of layers, printed live: `(3, 5)` → `(3, 5, 100)` → packed `(9, 100)` → `(1, 3, 128)` → `(3, 128)` → `(3, 2)`, and the observation that 591 600 of the 709 618 parameters are the embedding table

### 09.09.2026

Added:

- `code_files/lesson6.py` — the Lecture 6 script: attention and Transformers written out rather than taken from `nn.Transformer`. Scaled dot-product attention, multi-head splitting, sine/cosine positional encoding, pre-norm encoder blocks with residuals, masked mean pooling, and `--attention-map` to print the learned weights over real reviews. Runs both KazSAnDRA tasks on CPU with a fixed seed
- `language/en/06-attention-transformers.html`, `language/kk/06-attention-transformers.html`, `language/ru/06-attention-transformers.html` — Lecture 6, Attention and Transformers. What attention is (query/key/value, with the four steps computed by hand and checked against `F.scaled_dot_product_attention`), why the √d scaling is not optional (entropy 1.879 → 0.359 without it), the four types of attention with a diagram, the padding mask measured (59% of attention wasted without it), multi-head splitting, the measured proof that attention is order-blind and how positional encoding fixes it, the encoder block, and the full classifier
- Results on our own data: Transformer from scratch reaches **0.750** on polarity — below the LSTM's 0.772 and TF-IDF's 0.780 — and the lecture shows this is data starvation rather than tuning: 10 epochs gives 0.749, a smaller model 0.750, five times the data **0.787**. The five-class task lands at 0.378 against the LSTM's 0.374. Attention maps over real reviews show one crude global "find the sentiment word" pattern, which is what 8 000 reviews support

Changed:

- `page/js/course.js` — Lecture 6 published as "Attention and Transformers"
- `index.html` — link to Lecture 6

### 09.09.2026 (later)

Added:

- `tools/check-figures.py` — a layout checker for the inline SVG figures. It measures every `<text>` node and reports overlapping labels and text running past the viewBox, which is how the problems below were found rather than by eye
- Figures where there were none: **Lecture 0** gains 2 (what NLP sits between; the four reasons language is hard), **Lecture 1** gains 6 (the engine walking the text, character classes, quantifiers with greedy vs lazy, alternation and why brackets matter, anchors as zero-width positions, escaping), **Lecture 2** gains 6 (the whole pipeline and what each arrow discards, three tokenizers on one sentence, stemming vs lemmatization, the Bag of Words matrix with two identical rows, TF-IDF as frequency × rarity, n-grams keeping negation)

Fixed:

- Overlapping and overflowing text in five existing figures — `fig-space-t` in Lecture 4, `fig-state-t` and `fig-shapes-t` in Lecture 5, `fig-steps-t` and `fig-types-t` in Lecture 6 (the last two redrawn as a proper 2×2 grid)
- **Figure text was English on the Russian and Kazakh pages.** 313 labels across Lectures 3–6 were localised, plus every label and caption in the 14 new figures. Formulas (`hₜ = tanh(W·xₜ + U·hₜ₋₁ + b)`, `softmax(Q·Kᵀ / √d) · V`) are deliberately left as they are

All 96 inline SVGs are valid XML, the layout checker reports zero problems, and figure counts match across en/kk/ru in every lecture.

### 09.09.2026 (rewrite)

Changed:

- `language/*/06-attention-transformers.html` — Lecture 6 rebuilt as a full step-by-step course on attention and Transformers, roughly three times its previous size (34 sections, 30 inline SVG figures per language, up from 16 and 5). The measured KazSAnDRA results are unchanged and still come from `code_files/lesson6.py`; what was added is the teaching path around them:
  - **Before attention** — the 2014 encoder–decoder RNN drawn in full ("The cat sat on the mat" → "Кошка сидела на коврике"), the fixed context vector as a bottleneck, and Bahdanau's fix as a weighted read over all encoder states
  - **Intuition first** — one running example ("The animal didn't cross the street because *it* was tired"), the weights *it* computes over the sentence, and the seven questions (problem / intuition / inputs / operation / output / why / where) that every later component is introduced with
  - **Q, K, V separately** — the three roles as cards, the database analogy and its limit, `Q = XW_Q` with every symbol's shape spelled out, and a projection diagram
  - **A worked example with exact arithmetic** — three Kazakh tokens in two dimensions, carried through X → Q/K/V → `QKᵀ` → `÷√2` → softmax → `·V` as four `.matrix` tables, ending with «жақсы» moving from `[0, 1]` to `[1.436, 2.152]`. Every number verified independently
  - **Scaling argued, not asserted** — why dot products grow like √d, why softmax is scale-sensitive, why a saturated softmax has no gradient, then the existing entropy measurement (1.879 → 0.359)
  - **The variants taught, not listed** — additive (Bahdanau) vs multiplicative (Luong) vs scaled dot-product on one axis, and self / causal / cross / padding on the other, with a comparison table of where Q, K and V come from
  - **Causal masking** — the triangular mask over "I love deep learning", why −inf rather than 0, and why training stays parallel
  - **Multi-head** — four simultaneous relationships in one sentence, the eight-head diagram, and the full shape table at `d_model = 512`, `h = 8`, `d_k = 64`
  - **The Transformer itself** — a large encoder–decoder diagram as the central visual, then decomposed: tokenization → ids → embeddings → positions, the encoder block component by component, a dedicated figure for "attention mixes across tokens, the FFN transforms within one", residuals and LayerNorm, and what changes with depth
  - **The decoder** — the three-sub-layer block, cross-attention with a rectangular EN→KK alignment map, an eleven-step walkthrough of "I love machine learning" → "Мен машиналық оқытуды жақсы көремін", and the output head on "The capital of Kazakhstan is ___"
  - **The three families** — encoder-only, decoder-only, encoder–decoder side by side, and why decoder-only won
  - **Modern LLMs** — pre-norm, RMSNorm, SwiGLU, RoPE, GQA/MQA and the KV cache, framed strictly as "how 2024 differs from 2017"
  - **Code mapped to maths** — scaled dot-product attention, `MultiHeadAttention`, `Block` and the full model, each explained at block level with a figure linking code lines to diagram boxes
  - **A tensor-shape reference table** and a closing "one token, all the way through" section that answers the fifteen questions the lecture must leave a student able to answer
  - Knowledge check grown from 5 to 8 questions, practice tasks from 10 to 11 (including building a tiny causal language model out of the classifier), quiz rewritten
- `page/js/course.js` — Lecture 6 topics line updated to "Attention · Q/K/V · Multi-head · Encoder & decoder · PyTorch"

All 90 inline SVGs (30 per language) are valid XML, `tools/check-figures.py` reports zero layout problems, figure and section counts match across en/kk/ru, and every code block is byte-identical in all three languages

### 14.09.2026

Added:

- `code_files/lesson7.py` — the Lecture 7 script: fine-tuning pretrained encoders on the same balanced KazSAnDRA splits as Lectures 3–6, plus SST-2. `balance()` is copied unchanged from Lecture 3 so every number stays comparable. A hand-written training loop rather than `Trainer`, with `--compare` to run a model list under one identical recipe, `--frozen` for feature extraction, `--mlm-demo` for fill-in-the-blank, `--tokenizer-demo` for sub-word splitting, and `--train-size` for the data-efficiency sweep. Runs on Apple MPS
- `language/en/07-bert-finetuning.html`, `language/kk/07-bert-finetuning.html`, `language/ru/07-bert-finetuning.html` — Lecture 7, BERT and Fine-tuning. 21 sections and 14 inline SVG figures per language: why 8 000 labels cannot teach a language and a task at once, pretrain-once/fine-tune-many, masked language modelling with the 15% and 80/10/10 recipes and why NSP was dropped, BERT as Lecture 6's encoder stack plus `[CLS]`/`[SEP]`/segments, what fine-tuning adds and changes, the family (RoBERTa, DistilBERT, TinyBERT, MiniLM, ALBERT, ELECTRA, DeBERTa, mBERT, XLM-R) with a knowledge-distillation figure, how to choose a model, and what pretraining still does not give you

Measured, all on one fixed recipe (8 000 balanced training rows, 3 epochs, lr 2e-5, batch 32, max length 64, Apple M4 Pro / MPS):

- **Before any fine-tuning.** Fill-in-the-blank over three models: mBERT puts the capital of Kazakhstan at Алматы 0.31 and cannot produce a Kazakh verb at all; XLM-R gets Астана 0.62; only `kaz-roberta-conversational` produces correct first-person agreement (`оқыдым` 0.35, `оқимын` 0.15). Tokenizers on `кітаптарымыздан`: mBERT 6 pieces (tearing the leading «к» off the root), MiniLM and XLM-R 4 (the correct morphological split), kaz-RoBERTa 2
- **Polarity, 2 classes.** mBERT 0.799, XLM-R 0.797, kaz-RoBERTa 0.794, DistilmBERT 0.791, MiniLM 0.789 — against Lecture 6's from-scratch Transformer at 0.750. Pretraining is worth **+0.049**; the spread *between* pretrained models is 0.010, which is a tie. After one epoch the ranking is exactly what the two diagnostics predicted (kaz-RoBERTa 0.800, XLM-R 0.792, MiniLM 0.780, mBERT 0.762) — so they predict how fast a model arrives, not where it lands
- **Score, 5 classes.** kaz-RoBERTa 0.381, mBERT 0.377, DistilmBERT 0.373, XLM-R 0.370, MiniLM 0.358 — against Lecture 6's 0.378 achieved on 19 600 rows. Here pretraining bought **data efficiency, not accuracy**: the same result on 2.45× less data. The spread widens to 0.023, and the ceiling is label noise, exactly as the Lecture 5 and 6 confusion matrices showed
- **SST-2, English, same 8 000 rows.** RoBERTa 0.937, ALBERT 0.917, BERT 0.915, DistilBERT 0.877, ELECTRA-small 0.855. Four published claims reproduced on our own numbers: RoBERTa beats BERT (+0.022); ALBERT matches BERT with 9.4× fewer parameters but is only 1.45× faster; ELECTRA-small is 5.8× faster and 8.1× smaller; DistilBERT retains 95.8%, slightly under the advertised ~97%. **0.937 English against 0.799 Kazakh on identical data — 3.2× as many errors**, which is a fact about pretraining corpora rather than about the languages
- **Frozen encoder vs full fine-tuning.** mBERT frozen 0.558 (its loss never left 0.691, against ln 2 = 0.693 for random) versus 0.799 fine-tuned; kaz-RoBERTa frozen 0.741 versus 0.794. Freezing is 5–6× faster. The two rows are not directly comparable — BERT's head is 1 538 weights and RoBERTa's is 592 130, a 385× confound that is stated on the page
- **How few labels are needed.** kaz-RoBERTa on polarity: 250 rows 0.685, 500 0.720, 1 000 0.766, 2 000 0.780, 4 000 0.787, 8 000 0.794. **2 000 labels with a pretrained model beat the 8 000 that TF-IDF and the from-scratch Transformer needed** — a quarter of the annotation budget for a better result

Fixed:

- `code_files/lesson7.py` — freezing the encoder made PyTorch select the fused MPS attention kernel, which raises `NotImplementedError: scaled_dot_product_attention for MPS does not support dropout`. A frozen encoder is a fixed feature function and should be deterministic anyway, so it is now put in `eval()` mode during training, which is both the correct semantics and the fix

Changed:

- `code_files/requirements.txt` — `transformers`, `sentencepiece` and `accelerate` added
- `page/js/course.js` — Lecture 7 published as "BERT and Fine-tuning" / "BERT және Fine-tuning" / "BERT и Fine-tuning"
- `index.html` — link to Lecture 7

All 42 inline SVGs across the three Lecture 7 pages are valid XML, `tools/check-figures.py` reports zero layout problems, section and figure counts match across en/kk/ru, and every code block is byte-identical in all three languages

### 16.09.2026

Added:

- `code_files/lesson8.py` — the Lecture 8 script: every number on the page measured on a laptop, with **no training anywhere**. `--tokens` (tiktoken and Qwen tokenizers over one parallel en/ru/kk message, plus an API bill), `--memory` (parameters → gigabytes, and KV-cache arithmetic), `--sampling` (greedy, temperature, top-k, top-p and the entropy of the distribution behind them, on `Qwen/Qwen2.5-0.5B-Instruct`), `--lora` (counts LoRA's trainable parameters with `peft`, does not train them), `--quantize` (round real weights to int8 and compare the generation). CPU/MPS, fixed seeds
- `language/en/08-large-language-models.html`, `language/kk/08-large-language-models.html`, `language/ru/08-large-language-models.html` — Lecture 8, Large Language Models. 16 sections and 11 inline SVG figures per language: the inference loop as Lecture 6's decoder repeated, the three training stages (pretraining / instruction tuning / preference tuning) and what each one buys, parameter counts, open-weight vs open-source vs closed API, tokenizer economics, how to choose a model, sampling, reasoning models, RLHF/DPO/RLVR and reward hacking, LoRA and quantization

Measured (Apple M4 Pro, 24 GB unified memory, `Qwen/Qwen2.5-0.5B-Instruct` where a model is needed):

- **Tokens are not words, and a language has a price.** One message, same meaning, three languages. Under `o200k_base`: en 19, ru 25, kk 38 tokens — **Kazakh costs 2.0× English**. Under the older `cl100k_base` it is 4.53×, and under Qwen2.5 3.79×. Per word: 1.19 tokens for English against 3.17 for Kazakh on the best tokenizer tested. The choice of tokenizer changes a Kazakh product's bill by more than a factor of two
- **An API bill, computed rather than guessed.** 100 000 calls a month, both token counts scaled by the measured per-language multiplier. A chat-shaped workload (20 words in, 100 out) on `claude-sonnet-5`: **\$130 English, \$193 Russian, \$336 Kazakh**. A classification-shaped workload (2 000 words in, 5 out) on the same model: \$488 / \$730 / \$1 289. The Kazakh penalty is ~2.6× in both shapes, and the workload shape moves the absolute bill by 3.8×. Anthropic's rates are current as of this date; other vendors' are labelled as an example of the method, not a quote
- **Parameters to gigabytes.** 7 B needs 28 GB in fp32, 14 GB in bf16, 3.5 GB in int4; 70 B needs 140 GB in bf16 and does not fit on this machine at any precision. **The KV cache is the constraint people miss**: an 8 B model with 32 layers and 8 KV heads spends 128 KB per token per sequence, so one 128 000-token conversation needs 16.78 GB — more than the weights — and 100 concurrent users at that length need 1.68 TB
- **Sampling, on a real model.** Greedy is bit-identical across seeds; at `temperature=0.7` two seeds give two different sentences; at 1.5 the output degenerates into cross-lingual noise. The distribution behind it, printed: entropy 0.00 at T=0.1, 0.14 at T=1.0, 8.31 at T=2.0, with the top token falling from 1.000 to 0.188. Temperature adds no knowledge — it only reshapes one distribution
- **LoRA, counted and not trained** (the user has no GPU, so the fine-tuning sections show code and arithmetic only). On a 494 M-parameter model, LoRA trains **540 672 parameters at rank 4 (0.109%)** and 8 650 752 at rank 64 (1.751%); the adapter is 2.2–34.6 MB against ~1 GB for the base model. The maths spelled out: at d = k = 896 and r = 8 you train 14 336 numbers instead of 802 816, and `B·A` folds back into `W` so inference costs nothing extra
- **Quantization, with the error measured.** int8 on one real (896, 896) weight matrix: mean absolute error 0.000484, worst single error 0.004829, **1.15% relative error**, 4 209 distinct values collapsed to 255. Rounding all 169 linear layers to int8 and back left the generation **byte-identical** to fp32, at a quarter of the memory
- **What a small model gets wrong about Kazakhstan.** The 0.5 B model repeatedly placed Almaty "near the Caspian Sea" and called it the capital. The lecture keeps this rather than hiding it: it is the section on why anything specific, recent or thinly represented has to be supplied to a model rather than recalled from it

Changed:

- `code_files/requirements.txt` — `tiktoken` and `peft` added
- `page/js/course.js` — Lecture 8 published as "Large Language Models"
- `index.html` — link to Lecture 8

All 33 inline SVGs across the three Lecture 8 pages are valid XML, `tools/check-figures.py` reports zero layout problems, section and figure counts match across en/kk/ru, and all 18 code blocks are byte-identical in all three languages

### 21.09.2026

Added:

- `code_files/lesson9/` — the Lecture 9 package, and the first lecture whose model runs locally rather than being described. `serve.sh` downloads `unsloth/Qwen3.5-4B-GGUF` (Q4_K_M, 2.7 GB) and starts `llama-server` with `--jinja` and `--n-gpu-layers 99`; everything else talks to it through the ordinary OpenAI client at `http://127.0.0.1:8080/v1`, so the same code works against vLLM, Ollama or a paid endpoint with one line changed
  - `llm.py` — `ask`, `chat`, `stream`, with the reasoning switch passed as `extra_body={"chat_template_kwargs": {"enable_thinking": …}}` and thinking split out of the reply
  - `techniques.py` + `app.py` — the Gradio lab on port 7860: **Chat** (streaming, thinking in its own panel), **Techniques** (nine prompt-engineering techniques, each run the naive way and the engineered way side by side with tokens and seconds, every prompt editable in the page), **RAG** (retrieval on/off, BM25/dense/hybrid, k, with the retrieved passages shown)
  - `corpus.py`, `retrieve.py`, `rag.py` — RAG over the course's own pages: 1 924 passages in three languages, BM25 written out in 30 lines, dense embeddings from `multilingual-e5-small`, and reciprocal rank fusion
  - `prompting.py`, `tasks.py` — optional batch scoring of the same techniques over hundreds of examples
- `language/en/09-prompting-rag.html`, `language/kk/09-prompting-rag.html`, `language/ru/09-prompting-rag.html` — Lecture 9, Prompt Engineering and RAG. 22 sections and 10 inline SVG figures per language: three places to change a model's behaviour and the order to try them, serving a GGUF, what the chat template actually sends, reasoning on and off, seven measured prompt techniques, then chunking, embedding, three retrievers, the prompt RAG assembles, and what RAG still does not fix

Measured (Apple M4 Pro, Qwen3.5-4B at Q4_K_M, served by llama.cpp):

- **Reasoning on against off**, twelve checkable questions, only `enable_thinking` changing: direct **11/12** at 225 output tokens and 4.0 s; thinking **10/12** at 1 683 output tokens (1 541 of them thinking) and 30.3 s. **7.5× the tokens and 7.6× the time for no gain** on problems the model already solves in one pass. One of the two thinking losses was an *empty* reply, not a wrong one
- **A prompt with no training ties a fine-tuned encoder.** On the first 400 rows of the exact KazSAnDRA test split Lecture 7 used — same file, same `balance()`, same seed 42 — a bare question scores 0.708, `role + format` **0.790**, `+ rubric` 0.790, `few-shot (8)` 0.795. Lecture 3's TF-IDF got 0.792 and Lecture 7's fine-tuned mBERT 0.799. The bare prompt is the most accurate *per answer* (0.835) and the least usable: **61 of 400 replies could not be parsed at all**, against 0 for the engineered prompt
- **Stating a JSON schema beats asking for JSON.** Fifteen calls each: "extract it as JSON" gave 0/15 directly parseable replies (all fenced) and **0/15 with valid values** — the model invented topics like `"app update"`; the spelled-out schema gave 15/15 on every column. Adding a worked example on top changed nothing
- **What a system prompt buys.** Asked in Kazakh, all three prompts answered in Kazakh 8/8 — the system prompt bought *length*, 85 words down to 24. Asked in English or Russian, `0/8` replies contained a Kazakh-specific letter without the instruction and 8/8 with it
- **Prompt injection**: undefended **5/5 hijacked**, delimited 2/5, delimiters plus an instruction hierarchy **0/5**. Stated on the page as a cost increase, not a boundary
- **Self-consistency**: five samples, majority vote, 11/12 — *identical* to one sample at 5× the tokens, because the five samples agreed. Published rather than hidden
- **Retrieval**, 1 924 chunks, ten questions with gold phrases verified to appear in only 3–13 chunks each: at k = 5 all three retrievers score 10/10 and the comparison says nothing; at **k = 1 BM25 gets 5/10 against dense 8/10**, and mean rank is 1.9 / 1.3 / 1.2 for BM25 / dense / hybrid
- **Closed book 0/10, with retrieval 10/10**, for 60 → 1 639 prompt tokens (**27.1×**). The facts were measured in Lectures 3–8 and exist nowhere else, so 0/10 is the correct closed-book result
- **Refusal 3/4.** The failure is the teaching moment: asked for the instructor's phone number, the system retrieved Lecture 1's regex example and answered `+7 701 123 45 67` **with a citation**

Fixed, and kept on the page as method:

- `rag.py` — a section named `k` in the `SECTIONS` dict auto-created a `--k` flag that collided with the `-k` value argument, so `args.k` was `False` and **every RAG prompt was built with zero passages**. The whole grounding run scored 0/10 with ten "honest refusals" that were in fact correct behaviour on an empty context. Renamed to `--passages`, with an assertion so the collision cannot come back
- `retrieve.py` — the evaluation's gold phrases were originally the answers themselves ("6", "2", "0.79"). All three retrievers scored 10/10 because "6" appears in 871 of 1 924 chunks and "2" in 1 294. Replaced with phrases checked to appear in 3–13 chunks, and `--gold-frequency` added so the check is part of the tool
- Two grader bugs of our own, both reported on the page: a substring match accepted a closed-book reply saying "128 **bytes** per token" for an answer of 128 KB, and one question's expected answer was simply wrong — the entropy without √d scaling is 0.359, not 1.879. Correcting both moved closed book from 1/10 to 0/10 and retrieval from 9/10 to 10/10
- The reasoning experiment first ran at `max_tokens=2048` and reported 0.58 for thinking against 0.92 for direct. That was truncation, not accuracy: several runs stopped inside `<think>` and returned empty strings. The run now reports how many replies hit the cap, and the page shows the wrong number and the correction

Changed:

- `code_files/requirements.txt` is unchanged; `code_files/lesson9/requirements.txt` adds `openai`, `gradio>=6.0`
- `page/js/course.js` — Lecture 9 published as "Prompt Engineering and RAG"
- `index.html` — link to Lecture 9

All 30 inline SVGs across the three Lecture 9 pages are valid XML, `tools/check-figures.py` reports zero layout problems, section and figure counts match across en/kk/ru, and all 33 code blocks are byte-identical in all three languages

### 21.09.2026 (layout fix)

Fixed, after the Lecture 9 pages were reported as visually broken:

- **`language/*/09-prompting-rag.html` used the wrong markup for the `breakdown` component.** `.breakdown__list` is a CSS grid styled for a `<dl>` with `<dt>`/`<dd>` children; the new pages used `<ul>` with `<li><strong>…</strong><span>…</span></li>`. Every `<li>` therefore became a grid item in the `max-content` column and sized to its full unwrapped width, so text ran outside every panel and the page scrolled sideways: **3 230 px wide in a 1 400 px window**. All 13 lists per language converted to `<dl>`; the page is now exactly viewport width
- `tools/check-render.py` — a new checker that renders each page in headless Chrome and measures `getBBox()` against the enclosing `<rect>`, plus `scrollWidth` against `clientWidth`. The existing `check-figures.py` estimates text width from character counts, which under-measures Cyrillic and cannot see CSS at all — it reported zero problems on a page that was visibly broken. Run it as `python3 tools/check-render.py --widths 1920,1400,900,420`
- **43 SVG labels across Lectures 0–8 were spilling out of their boxes** and had never been caught, almost all on the Kazakh and Russian pages. 27 fixed by reducing font-size by 0.5–1.0, 16 by shortening the label (for example `ойлау: жоспар, талпыныс, тексеру, кері қайту` → `ойлау: жоспар, талпыныс, тексеру`)
- `page/css/styles.css` — `.breakdown__list` now uses `grid-template-columns: minmax(0, max-content) 1fr` and `dt` no longer sets `white-space: nowrap`, so a long label wraps instead of pushing the page sideways. This was making Lecture 7 scroll horizontally at 900 px in Kazakh and Russian. A rule was also added so long identifiers inside narrow cards break rather than widen the card
- `language/*/05-rnn-lstm.html` — `embedding.weight.requires_grad = False` was bare text in a narrow card and overflowed it in all three languages; now wrapped in `<code>` like every other identifier in the course

Verified: 30 pages × 5 viewport widths (1920, 1400, 900, 620, 420) — **0 label spills, 0 element overflows, 0 pages scrolling horizontally**. All 276 inline SVGs are valid XML, tags balance on every page, and code blocks remain byte-identical across en/kk/ru in Lectures 6–9.

### 24.09.2026

Added:

- `code_files/lesson10/` — the Lecture 10 agent: a travel assistant for Kazakhstan that **knows nothing**. Every distance, forecast, exchange rate and description in its answers is fetched with a tool at run time, from public APIs that need no key — which is both what an agent is and how the lecture avoids stating a single unsourced fact about Kazakhstan
  - `tools.py` — six tools with JSON schemas and a dispatcher: `geocode`, `distance_km`, `weather_forecast`, `wikipedia` (en/kk/ru), `convert_currency`, `calculate`. Sources: Open-Meteo geocoding and forecast, Wikipedia REST, open.er-api.com
  - `agent.py` — the loop in about twenty lines, plus a trace, a step limit (`MAX_STEPS = 8`) and per-turn token accounting
  - `llm.py` — reads `.env` and picks the backend: OpenAI when `OPENAI_API_KEY` is set, Lecture 9's local `llama-server` otherwise. The agent code never learns which it got
  - `app.py` — Gradio on port 7861, with the trace shown beside the answer, green for a tool that worked and red for one that failed
- `language/en/10-ai-agents.html`, `language/kk/10-ai-agents.html`, `language/ru/10-ai-agents.html` — Lecture 10, AI Agents. 16 sections and 7 inline SVG figures per language: pipeline against agent, the loop, the message protocol, a tool as function + schema + description, a full trace, recovery from a failed tool, refusal, parallel calls, what a loop costs, the instruction the agent ignored, six failure modes, guardrails, and when not to build one

Measured (`gpt-4o-mini`, 24 September 2026, seven real runs):

| question | tool calls | turns | prompt tokens | seconds |
|---|---|---|---|---|
| weather on 1 March next year (refused) | 0 | 1 | 572 | 2.4 |
| 500 USD for 3 days in Shymkent | 2 | 2 | 1 313 | 3.0 |
| two days in Almaty, what to pack | 2 | 2 | 1 397 | 4.5 |
| Astana–Almaty distance and weather | 2 | 2 | 1 574 | 11.6 |
| compare Almaty and Shymkent | 4 | 2 | 2 052 | 6.8 |
| Charyn Canyon (recovered from 2 failures) | 5 | 5 | 4 751 | 12.9 |
| 300 000 ₸ for 4 days in Turkestan | 7 | 7 | 6 451 | 17.2 |

- **The same agent used 0, 2, 4, 5 and 7 tool calls** on five ordinary questions. That unpredictability *is* the definition — and it means an 11× spread in cost and a 2.4 s to 17.2 s spread in latency for questions a user would consider equivalent
- **Turns cost, not calls.** 4 calls batched into 2 turns cost 2 052 prompt tokens; 5 calls chained across 5 turns cost 4 751. The history is resent once per turn, so the longest run went 588 → 669 → 728 → 822 → 1 055 → 1 274 → 1 315
- **It recovered from its own broken tool.** Asked about Чарынский каньон, `distance_km` failed on the Cyrillic name; the model read the error message — *"try the Latin spelling, for example Almaty, Astana, Shymkent"* — geocoded in Latin and finished. It also reissued an identical failing call it could still see in its own history, which is why the step limit exists
- **The cheapest correct answer used no tools at all.** Asked for a forecast five months out it refused, in 572 tokens, instead of calling the tool with a nonsense date and paraphrasing whatever came back
- **It disobeyed its own system prompt, and the answer was still right.** Told to use `calculate` for every sum, it called the tool for `500/3` and then produced 74 146.34 — `222439.02 / 3` — in its head. Correct to the cent, and invisible outside the trace. Written up as its own section, because a silent instruction failure is the one you ship

Fixed, and kept on the page as method:

- `tools.py` — the first `geocode` searched the gazetteer in **English only**, so a Kazakh question spent four of eight steps failing to find its own cities (`Астана`, `Алматы`, `Нұр-Сұлтан` all returned "not found"). It now tries English, Russian and Kazakh, then transliterates Kazakh-specific letters, then consults a small alias table, and prefers populated places over airports and mountain peaks. Same model, same prompt, half the steps. The page uses this to make the point that a bad trace is often the tool's fault, not the model's
- `tools.py` — added retries on 429/5xx after a real Open-Meteo 503 landed mid-run
- `llm.py` — `.env` is now searched in three places (`lesson10/`, `code_files/`, repo root) instead of only the repo root, which is where the key actually was

Tested without spending anything: the six tools against the live APIs, and the loop against a scripted model covering parallel calls, a hallucinated tool name, malformed JSON arguments, wrong argument names, the step limit, and `calculate("__import__('os').system('echo pwned')")` — refused with `only + - * / // % ** and numbers are allowed`, because the expression is parsed by an AST walker and never `eval`-ed.

Changed:

- `page/js/course.js` — slot 10 published as "AI Agents" (was a placeholder for "Advanced NLP", whose LLM and fine-tuning material is already covered by Lectures 7 and 8)
- `index.html` — link to Lecture 10

All 21 inline SVGs across the three Lecture 10 pages are valid XML, `tools/check-figures.py` and `tools/check-render.py` both report zero problems at 1920, 1400, 900 and 420 px, section and figure counts match across en/kk/ru, and all 9 code blocks are byte-identical in all three languages.
