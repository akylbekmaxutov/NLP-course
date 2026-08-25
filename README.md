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
