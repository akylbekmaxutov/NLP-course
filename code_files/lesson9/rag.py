"""RAG: retrieval-augmented generation over this course's own pages.

Prompting changes how the model uses what it knows. RAG changes what it knows,
at the moment of the question, without touching a single weight.

    python3 corpus.py            # 1. cut the pages into passages
    python3 retrieve.py --build  # 2. embed them once
    ./serve.sh                   # 3. terminal 1
    python3 rag.py --all         # 4. terminal 2
"""
import argparse
import re
import statistics

import llm
import retrieve

SYSTEM = (
    "You answer questions about an NLP course using only the numbered passages "
    "you are given.\n"
    "Rules:\n"
    "- Use only the passages. Do not use anything you remember from elsewhere.\n"
    "- Cite the passage you used as [1], [2] and so on.\n"
    "- If the passages do not contain the answer, reply exactly: "
    "NOT IN THE MATERIAL.\n"
    "- Be brief: one or two sentences."
)

TEMPLATE = "Passages:\n\n%s\n\nQuestion: %s"

CLOSED_BOOK_SYSTEM = (
    "You answer questions about an NLP course. Be brief: one or two sentences. "
    "If you do not know, say so."
)


def answer(question, retriever, k=5, method="hybrid", think=False,
           max_tokens=300):
    hits = retriever.search(question, k=k, method=method)
    prompt = TEMPLATE % (retrieve.format_context(hits), question)
    reply = llm.ask(prompt, system=SYSTEM, think=think, max_tokens=max_tokens,
                    temperature=0.3, seed=11)
    return reply, hits


def closed_book(question, think=False, max_tokens=300):
    return llm.ask(question, system=CLOSED_BOOK_SYSTEM, think=think,
                   max_tokens=max_tokens, temperature=0.3, seed=11)


def rule(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def contains(text, needle):
    return needle.lower() in (text or "").lower()


# ==========================================================================
# 1. The same questions, with and without the passages
# ==========================================================================
def experiment_grounding(retriever, k=5):
    rule("1. CLOSED BOOK vs RETRIEVAL")
    print("  Ten questions whose answers were measured in Lectures 3-8 and")
    print("  exist nowhere else. The model cannot have seen them.\n")

    closed_hits = rag_hits = refusals = 0
    in_tokens = []
    for question, gold, needle in retrieve.QUESTIONS:
        bare = closed_book(question)
        grounded, hits = answer(question, retriever, k=k)

        closed_ok = contains(bare.text, needle)
        rag_ok = contains(grounded.text, needle)
        closed_hits += closed_ok
        rag_hits += rag_ok
        refusals += "NOT IN THE MATERIAL" in grounded.text.upper()
        in_tokens.append((bare.prompt_tokens, grounded.prompt_tokens))

        print("  Q: %s" % question[:68])
        print("     expected  %s" % needle)
        print("     closed    %-5s %s" % ("ok" if closed_ok else "MISS",
                                          bare.text.replace("\n", " ")[:88]))
        print("     +RAG      %-5s %s" % ("ok" if rag_ok else "MISS",
                                          grounded.text.replace("\n", " ")[:88]))
    n = len(retrieve.QUESTIONS)
    print("\n  closed book        %d/%d correct" % (closed_hits, n))
    print("  with retrieval     %d/%d correct  (%d honest refusals)"
          % (rag_hits, n, refusals))
    print("  prompt size        %.0f tokens -> %.0f tokens  (%.1fx)"
          % (statistics.mean(a for a, _ in in_tokens),
             statistics.mean(b for _, b in in_tokens),
             statistics.mean(b for _, b in in_tokens) /
             max(1, statistics.mean(a for a, _ in in_tokens))))


# ==========================================================================
# 2. Does the retriever choice change the answer?
# ==========================================================================
def experiment_methods(retriever, k=5):
    rule("2. BM25 vs DENSE vs HYBRID, end to end")
    print("  the retriever is the only thing that changes\n")
    print("  %-22s %14s %14s" % ("retriever", "passage found", "answer correct"))
    print("  " + "-" * 54)
    for method in ("bm25", "dense", "hybrid"):
        found = correct = 0
        for question, gold, needle in retrieve.QUESTIONS:
            hits = retriever.search(question, k=k, method=method)
            found += any(gold in c["text"] for c, _ in hits)
            reply, _ = answer(question, retriever, k=k, method=method)
            correct += contains(reply.text, needle)
        n = len(retrieve.QUESTIONS)
        print("  %-22s %14s %14s"
              % (method, "%d/%d" % (found, n), "%d/%d" % (correct, n)))
    print("\n  Retrieval is the ceiling: the generator cannot answer from a")
    print("  passage that was never fetched.")


# ==========================================================================
# 3. How many passages to send
# ==========================================================================
def experiment_passages(retriever, values=(1, 3, 5, 10)):
    rule("3. HOW MANY PASSAGES")
    print("  %-6s %14s %14s %12s %10s"
          % ("k", "passage found", "answer correct", "prompt tok", "seconds"))
    print("  " + "-" * 62)
    for k in values:
        found = correct = 0
        tokens, seconds = [], []
        for question, gold, needle in retrieve.QUESTIONS:
            hits = retriever.search(question, k=k)
            found += any(gold in c["text"] for c, _ in hits)
            reply, _ = answer(question, retriever, k=k)
            correct += contains(reply.text, needle)
            tokens.append(reply.prompt_tokens)
            seconds.append(reply.seconds)
        n = len(retrieve.QUESTIONS)
        print("  %-6d %14s %14s %12.0f %10.1f"
              % (k, "%d/%d" % (found, n), "%d/%d" % (correct, n),
                 statistics.mean(tokens), statistics.mean(seconds)))
    print("\n  More passages raise the chance the answer is present and lower")
    print("  the chance the model reads the right one. That trade-off is why")
    print("  reranking exists.")


# ==========================================================================
# 4. Questions the corpus cannot answer
# ==========================================================================
UNANSWERABLE = [
    "What was the exam timetable for this course?",
    "How many students are enrolled in this course?",
    "What is the instructor's phone number?",
    "Which grade is needed to pass Lecture 7?",
]


def experiment_refusal(retriever, k=5):
    rule("4. WHEN THE ANSWER IS NOT THERE")
    print("  Four questions about this course that the pages do not answer.")
    print("  The only correct behaviour is to refuse.\n")
    refused = 0
    for question in UNANSWERABLE:
        reply, hits = answer(question, retriever, k=k)
        ok = "NOT IN THE MATERIAL" in reply.text.upper()
        refused += ok
        print("  %-5s %-52s %s" % ("ok" if ok else "MADE UP", question[:52],
                                   reply.text.replace("\n", " ")[:60]))
    print("\n  refused correctly: %d/%d" % (refused, len(UNANSWERABLE)))
    print("  Retrieval always returns its k nearest passages, however far away")
    print("  they are. Knowing when to refuse is the generator's job, and the")
    print("  instruction that asks for it is doing real work.")


# ==========================================================================
# 5. Asking in Kazakh and Russian about English material
# ==========================================================================
CROSS = [
    ("kk", "Attention дегеніміз не және ол қалай жұмыс істейді?"),
    ("kk", "Fine-tuning мен LoRA арасындағы айырмашылық қандай?"),
    ("ru", "Что такое TF-IDF и зачем он нужен?"),
    ("ru", "Чем BERT отличается от GPT?"),
]


def experiment_crosslingual(retriever, k=5):
    rule("5. CROSS-LINGUAL RETRIEVAL")
    print("  The index holds all three languages. A Kazakh question does not")
    print("  have to land on a Kazakh passage.\n")
    for lang, question in CROSS:
        hits = retriever.search(question, k=k)
        langs = [c["lang"] for c, _ in hits]
        reply, _ = answer(question, retriever, k=k)
        print("  [%s] %s" % (lang, question))
        print("       passages from: %s" % ", ".join(langs))
        print("       lectures      : %s"
              % ", ".join(str(c["lecture"]) for c, _ in hits))
        print("       answer        : %s" % reply.text.replace("\n", " ")[:100])
        print()


# ==========================================================================
# 6. One question, shown all the way through
# ==========================================================================
def walkthrough(question, retriever, k=4):
    rule("6. ONE QUESTION, ALL THE WAY THROUGH")
    print("  question: %s\n" % question)

    hits = retriever.search(question, k=k)
    print("  retrieved %d passages:" % len(hits))
    for n, (chunk, score) in enumerate(hits, 1):
        print("    [%d] %.4f  Lecture %d - %s  (%s)"
              % (n, score, chunk["lecture"], chunk["heading"][:40], chunk["source"]))

    context = retrieve.format_context(hits)
    prompt = TEMPLATE % (context, question)
    print("\n  the prompt that gets sent is %d characters:" % len(prompt))
    print("    system  %d chars" % len(SYSTEM))
    print("    context %d chars (%d passages)" % (len(context), len(hits)))
    print("    question %d chars" % len(question))

    reply = llm.ask(prompt, system=SYSTEM, think=False, max_tokens=300,
                    temperature=0.3, seed=11)
    print("\n  prompt tokens %d, output tokens %d, %.1f s"
          % (reply.prompt_tokens, reply.output_tokens, reply.seconds))
    print("\n  answer:\n    %s" % reply.text.replace("\n", "\n    "))

    cited = set(int(x) for x in re.findall(r"\[(\d+)\]", reply.text))
    print("\n  cited passages: %s" % (sorted(cited) or "none"))
    for n in sorted(cited):
        if 1 <= n <= len(hits):
            print("    [%d] -> %s" % (n, hits[n - 1]["source"]
                                      if isinstance(hits[n - 1], dict)
                                      else hits[n - 1][0]["source"]))


# ==========================================================================
SECTIONS = {
    "grounding": experiment_grounding,
    "methods": experiment_methods,
    "passages": experiment_passages,
    "refusal": experiment_refusal,
    "crosslingual": experiment_crosslingual,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in SECTIONS:
        ap.add_argument("--" + name, action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ask", help="one question, end to end")
    ap.add_argument("-k", "--top-k", type=int, default=5,
                    dest="top_k", help="passages per question")
    args = ap.parse_args()
    assert "k" not in SECTIONS, "a section named 'k' would collide with -k"

    llm.require_server()
    retriever = retrieve.Retriever()

    if args.ask:
        walkthrough(args.ask, retriever, k=args.top_k)
        return

    chosen = [n for n in SECTIONS if getattr(args, n)]
    if args.all or not chosen:
        chosen = list(SECTIONS)
    for name in chosen:
        if name == "passages":
            SECTIONS[name](retriever)
        else:
            SECTIONS[name](retriever, k=args.top_k)


if __name__ == "__main__":
    main()
