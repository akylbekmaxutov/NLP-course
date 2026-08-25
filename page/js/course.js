/* ==========================================================================
   NLP Course — course contents

   THIS IS THE ONLY FILE YOU EDIT WHEN YOU ADD A LECTURE.

   It feeds both the sidebar table of contents (every page) and the timeline
   on the roadmap page. To publish a new lecture:

     1. create the page in en/, kk/ and ru/ with the SAME file name
     2. find its entry below and set  file: "05-text-classification.html"
        (an entry with  file: null  is shown as "planned" and is not a link)

   Technical terminology stays in English in all three languages.
   ========================================================================== */

window.NLP_COURSE = {
  /* Interface strings used by the generated navigation. */
  ui: {
    en: {
      lecture: "Lecture",
      planned: "Planned",
      contents: "Contents",
      onThisPage: "On this page",
      tagline: "From Text Processing to Large Language Models"
    },
    kk: {
      lecture: "Дәріс",
      planned: "Жоспарда",
      contents: "Мазмұны",
      onThisPage: "Осы бетте",
      tagline: "Мәтінді өңдеуден Large Language Models дейін"
    },
    ru: {
      lecture: "Лекция",
      planned: "В плане",
      contents: "Содержание",
      onThisPage: "На этой странице",
      tagline: "От обработки текста до Large Language Models"
    }
  },

  /* Ordered list of every page in the course. */
  pages: [
    {
      id: "roadmap",
      file: "index.html",
      title: {
        en: "Course Roadmap",
        kk: "Курс жоспары",
        ru: "План курса"
      }
    },
    {
      id: "basics",
      num: 0,
      file: "00-basics-nlp.html",
      date: { en: "24 Aug 2026", kk: "24 тамыз 2026", ru: "24 августа 2026" },
      title: {
        en: "Basics of NLP",
        kk: "NLP негіздері",
        ru: "Основы NLP"
      },
      topics: {
        en: "What NLP is · Why language is hard · NLP in Kazakhstan",
        kk: "NLP дегеніміз не · Тіл неге қиын · Қазақстандағы NLP",
        ru: "Что такое NLP · Почему язык сложен · NLP в Казахстане"
      }
    },
    {
      id: "regex",
      num: 1,
      file: "01-regular-expressions.html",
      date: { en: "24 Aug 2026", kk: "24 тамыз 2026", ru: "24 августа 2026" },
      title: {
        en: "Regular Expressions",
        kk: "Regular Expressions",
        ru: "Regular Expressions"
      },
      topics: {
        en: "Patterns · Extraction · Cleaning · Regex playground",
        kk: "Үлгілер · Шығарып алу · Тазалау · Regex playground",
        ru: "Шаблоны · Извлечение · Очистка · Regex playground"
      }
    },
    {
      id: "text-processing",
      num: 2,
      file: "02-text-processing.html",
      date: { en: "26 Aug 2026", kk: "26 тамыз 2026", ru: "26 августа 2026" },
      title: {
        en: "Text Processing",
        kk: "Text Processing",
        ru: "Text Processing"
      },
      topics: {
        en: "Tokenization · Lemmatization · Bag of Words",
        kk: "Tokenization · Lemmatization · Bag of Words",
        ru: "Tokenization · Lemmatization · Bag of Words"
      }
    },
    {
      id: "classical-ml",
      num: 3,
      file: null,
      date: { en: "31 Aug 2026", kk: "31 тамыз 2026", ru: "31 августа 2026" },
      title: {
        en: "Classical Machine Learning for Text Data",
        kk: "Мәтін деректеріне арналған классикалық Machine Learning",
        ru: "Классический Machine Learning для текстовых данных"
      }
    },
    {
      id: "word-embeddings",
      num: 4,
      file: null,
      date: {
        en: "02 Sep 2026",
        kk: "02 қыркүйек 2026",
        ru: "02 сентября 2026"
      },
      title: {
        en: "Word Embeddings",
        kk: "Word Embeddings",
        ru: "Word Embeddings"
      },
      topics: { en: "word2vec", kk: "word2vec", ru: "word2vec" }
    },
    {
      id: "text-classification",
      num: 5,
      file: null,
      date: {
        en: "07 Sep 2026",
        kk: "07 қыркүйек 2026",
        ru: "07 сентября 2026"
      },
      title: {
        en: "Text Classification",
        kk: "Text Classification",
        ru: "Text Classification"
      }
    },
    {
      id: "transformers",
      num: 6,
      file: null,
      date: {
        en: "09 Sep 2026",
        kk: "09 қыркүйек 2026",
        ru: "09 сентября 2026"
      },
      title: {
        en: "Transformers and Attention Mechanism",
        kk: "Transformers және Attention механизмі",
        ru: "Transformers и механизм Attention"
      }
    },
    {
      id: "language-models-1",
      num: 7,
      file: null,
      date: {
        en: "14 Sep 2026",
        kk: "14 қыркүйек 2026",
        ru: "14 сентября 2026"
      },
      title: {
        en: "Language Models I",
        kk: "Language Models I",
        ru: "Language Models I"
      },
      topics: {
        en: "BERT · RoBERTa · Fine-tuning pretrained models",
        kk: "BERT · RoBERTa · Pretrained модельдерді Fine-tuning",
        ru: "BERT · RoBERTa · Fine-tuning предобученных моделей"
      }
    },
    {
      id: "language-models-2",
      num: 8,
      file: null,
      date: {
        en: "16 Sep 2026",
        kk: "16 қыркүйек 2026",
        ru: "16 сентября 2026"
      },
      title: {
        en: "Language Models II",
        kk: "Language Models II",
        ru: "Language Models II"
      },
      topics: {
        en: "Generative Models · InstructGPT · DialoGPT",
        kk: "Generative Models · InstructGPT · DialoGPT",
        ru: "Generative Models · InstructGPT · DialoGPT"
      }
    },
    {
      id: "prompting-rag-agents",
      num: 9,
      file: null,
      date: {
        en: "21 Sep 2026",
        kk: "21 қыркүйек 2026",
        ru: "21 сентября 2026"
      },
      title: {
        en: "Prompting, RAG, and Agents",
        kk: "Prompting, RAG және Agents",
        ru: "Prompting, RAG и Agents"
      }
    },
    {
      id: "advanced-nlp",
      num: 10,
      file: null,
      date: {
        en: "23 Sep 2026",
        kk: "23 қыркүйек 2026",
        ru: "23 сентября 2026"
      },
      title: {
        en: "Advanced NLP",
        kk: "Advanced NLP",
        ru: "Advanced NLP"
      },
      topics: {
        en: "Large Language Models · LLM Fine-tuning",
        kk: "Large Language Models · LLM Fine-tuning",
        ru: "Large Language Models · LLM Fine-tuning"
      }
    },
    {
      id: "speech",
      num: 11,
      file: null,
      date: {
        en: "28 Sep 2026",
        kk: "28 қыркүйек 2026",
        ru: "28 сентября 2026"
      },
      title: {
        en: "Speech Processing",
        kk: "Speech Processing",
        ru: "Speech Processing"
      },
      topics: { en: "ASR · TTS", kk: "ASR · TTS", ru: "ASR · TTS" }
    }
  ]
};
