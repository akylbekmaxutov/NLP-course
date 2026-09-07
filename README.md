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
