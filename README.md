# NLP-course

Static teaching website for a Natural Language Processing course, used by the
instructor as presentation and lecture material. Available in **English (en)**,
**Kazakh (kk)** and **Russian (ru)**.

Plain HTML, CSS and vanilla JavaScript — no build system, no dependencies.
Open `index.html` directly or publish the repository with GitHub Pages.

Every page carries the same **table of contents in the left sidebar**, so any
lecture is one click away from any other. There is no "next lecture" flow — the
sidebar is the navigation.

## Structure

```text
index.html          language selection + links to published lectures
css/styles.css      shared stylesheet for every page and language
js/course.js        the course contents — the only file to edit when adding a lecture
js/main.js          shared JavaScript (sidebar TOC, copy buttons, Regex playground, progress bar)

en/index.html                    course roadmap (English)
en/00-basics-nlp.html            Lecture 0
en/01-regular-expressions.html   Lecture 1
kk/…                             same filenames, Kazakh
ru/…                             same filenames, Russian

assets/images/
assets/icons/
```

Technical terminology (Tokenization, Transformer, Embedding, RAG, LLM, ASR,
TTS, …) stays in English in all three languages; explanatory sentences are
translated. Examples are Kazakhstan-first.

## Lectures

| # | Date | Lecture | Status |
|---|------|---------|--------|
| 0 | 24.08.2026 | Basics of NLP | published |
| 1 | 24.08.2026 | Regular Expressions | published |
| 2 | 26.08.2026 | Text Processing · Tokenization · Lemmatization · Bag of Words | planned |
| 3 | 31.08.2026 | Classical Machine Learning for Text Data | planned |
| 4 | 02.09.2026 | Word Embeddings · word2vec | planned |
| 5 | 07.09.2026 | Text Classification | planned |
| 6 | 09.09.2026 | Transformers and Attention Mechanism | planned |
| 7 | 14.09.2026 | Language Models I · BERT · RoBERTa · Fine-tuning | planned |
| 8 | 16.09.2026 | Language Models II · InstructGPT · DialoGPT | planned |
| 9 | 21.09.2026 | Prompting, RAG, and Agents | planned |
| 10 | 23.09.2026 | Advanced NLP · LLM Fine-tuning | planned |
| 11 | 28.09.2026 | Speech Processing · ASR · TTS | planned |

## Adding a lecture

1. Copy `en/01-regular-expressions.html` to the new file name (e.g.
   `02-text-processing.html`) and replace the content, reusing the existing
   component classes (`.hero`, `.section`, `.definition`, `.kz-example`,
   `.diagram`, `.timeline`, `.code-block`, `.callout`, `.takeaways`,
   `.regex-playground`). Update the three `lang-switch` links at the top of the
   page to point at the new file name.
2. Repeat for `kk/` and `ru/` keeping the same file name and section order.
3. In [js/course.js](js/course.js) find that lecture's entry and change
   `file: null` to `file: "02-text-processing.html"`.

That third step is all the navigation work there is: the sidebar contents on
every page and the timeline on the roadmap pages are both generated from
`js/course.js`. Section links inside a page are generated from its
`<h2 id="…">` headings, so no extra markup is needed for them either.

Authoring rules for the site live in [Guidelines.md](Guidelines.md).
