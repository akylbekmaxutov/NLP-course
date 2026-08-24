/* ==========================================================================
   NLP Course — shared JavaScript (vanilla ES6+, no dependencies)

   Features:
   - sidebar table of contents built from js/course.js
   - section links + scroll spy for the current page
   - sidebar drawer on small screens
   - copy-to-clipboard buttons on code blocks
   - interactive Regex playground with preset examples
   - reading progress indicator
   - remembering the last chosen language
   ========================================================================== */

(function () {
  "use strict";

  /* ------------------------------------------------------------------
     UI strings. Technical terminology stays in English on purpose.
     ------------------------------------------------------------------ */

  var STRINGS = {
    en: {
      copy: "Copy",
      copied: "Copied",
      copyFailed: "Copy failed",
      invalid: "Invalid Regular Expression",
      noMatches: "No matches",
      matches: function (n) {
        return n === 1 ? "1 match" : n + " matches";
      },
      emptyPattern: "Enter a pattern to see matches.",
      reset: "Reset",
      score: function (right, done, total) {
        return done ? right + " / " + done + " correct (" + done + " of " + total + " answered)" : "Answer the questions to see your score.";
      }
    },
    kk: {
      copy: "Көшіру",
      copied: "Көшірілді",
      copyFailed: "Көшіру мүмкін болмады",
      invalid: "Invalid Regular Expression",
      noMatches: "Сәйкестік табылмады",
      matches: function (n) {
        return n + " сәйкестік";
      },
      emptyPattern: "Сәйкестіктерді көру үшін pattern енгізіңіз.",
      reset: "Қайта бастау",
      score: function (right, done, total) {
        return done ? right + " / " + done + " дұрыс (" + total + " сұрақтың " + done + "-не жауап берілді)" : "Ұпайды көру үшін сұрақтарға жауап беріңіз.";
      }
    },
    ru: {
      copy: "Копировать",
      copied: "Скопировано",
      copyFailed: "Не удалось скопировать",
      invalid: "Invalid Regular Expression",
      noMatches: "Совпадений нет",
      matches: function (n) {
        return n + " совпадений";
      },
      emptyPattern: "Введите pattern, чтобы увидеть совпадения.",
      reset: "Сбросить",
      score: function (right, done, total) {
        return done ? right + " / " + done + " верно (отвечено " + done + " из " + total + ")" : "Ответьте на вопросы, чтобы увидеть результат.";
      }
    }
  };

  function lang() {
    var value = (document.documentElement.lang || "en").slice(0, 2);
    return STRINGS[value] ? value : "en";
  }

  function t() {
    return STRINGS[lang()];
  }

  function ui() {
    var course = window.NLP_COURSE;
    return (course && course.ui && course.ui[lang()]) || {};
  }

  var $ = function (sel, root) {
    return (root || document).querySelector(sel);
  };
  var $$ = function (sel, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(sel));
  };

  /* Current file name, e.g. "01-regular-expressions.html".
     Falls back to index.html for directory URLs like /en/. */
  function currentFile() {
    var parts = window.location.pathname.split("/");
    var last = parts[parts.length - 1];
    return last || "index.html";
  }

  /* ------------------------------------------------------------------
     Sidebar table of contents
     ------------------------------------------------------------------ */

  function initTableOfContents() {
    var host = $("[data-toc]");
    var course = window.NLP_COURSE;
    if (!host || !course) return;

    var code = lang();
    var labels = ui();
    var here = currentFile();
    var list = document.createElement("ol");
    list.className = "toc";

    course.pages.forEach(function (page) {
      var item = document.createElement("li");
      var isCurrent = page.file === here;
      var label = page.title[code] || page.title.en;
      var kicker =
        typeof page.num === "number"
          ? (labels.lecture || "Lecture") + " " + page.num
          : null;

      if (page.file) {
        var link = document.createElement("a");
        link.className = "toc__link";
        link.href = page.file;
        if (isCurrent) link.setAttribute("aria-current", "page");
        if (kicker) link.appendChild(span("toc__num", kicker));
        link.appendChild(document.createTextNode(label));
        item.appendChild(link);
      } else {
        var planned = document.createElement("span");
        planned.className = "toc__planned";
        if (kicker) planned.appendChild(span("toc__num", kicker));
        planned.appendChild(document.createTextNode(label));
        planned.appendChild(span("toc__badge", labels.planned || "Planned"));
        item.appendChild(planned);
      }

      if (isCurrent) {
        var sections = buildSectionList();
        if (sections) item.appendChild(sections);
      }

      list.appendChild(item);
    });

    host.innerHTML = "";
    host.appendChild(list);
    initScrollSpy();
  }

  function span(className, text) {
    var el = document.createElement("span");
    el.className = className;
    el.textContent = text;
    return el;
  }

  /* Section links for the page we are on, taken from the <h2 id="…"> headings. */
  function buildSectionList() {
    var headings = $$("main h2[id]");
    if (headings.length < 2) return null;

    var list = document.createElement("ul");
    list.className = "toc__sections";

    headings.forEach(function (heading) {
      var item = document.createElement("li");
      var link = document.createElement("a");
      link.href = "#" + heading.id;
      link.textContent = heading.textContent.trim();
      link.setAttribute("data-spy", heading.id);
      item.appendChild(link);
      list.appendChild(item);
    });

    return list;
  }

  /* Highlight the section currently on screen. */
  function initScrollSpy() {
    var links = $$("[data-spy]");
    if (!links.length) return;

    var targets = links
      .map(function (link) {
        return document.getElementById(link.getAttribute("data-spy"));
      })
      .filter(Boolean);

    var ticking = false;

    function update() {
      var offset = window.scrollY + 140;
      var activeIndex = 0;

      targets.forEach(function (target, index) {
        if (target.offsetTop <= offset) activeIndex = index;
      });

      links.forEach(function (link, index) {
        link.classList.toggle("is-active", index === activeIndex);
      });

      ticking = false;
    }

    window.addEventListener(
      "scroll",
      function () {
        if (!ticking) {
          window.requestAnimationFrame(update);
          ticking = true;
        }
      },
      { passive: true }
    );

    update();
  }

  /* ------------------------------------------------------------------
     Roadmap timeline (roadmap page only) — same source as the sidebar
     ------------------------------------------------------------------ */

  function initRoadmap() {
    var host = $("[data-roadmap]");
    var course = window.NLP_COURSE;
    if (!host || !course) return;

    var code = lang();
    var labels = ui();
    var list = document.createElement("ol");
    list.className = "timeline timeline--roadmap";

    course.pages.forEach(function (page) {
      if (typeof page.num !== "number") return; // skip the roadmap entry itself

      var item = document.createElement("li");
      if (page.file) item.className = "is-published";

      if (page.date) {
        item.appendChild(span("timeline__date", page.date[code] || page.date.en));
      }

      var title = document.createElement("p");
      title.className = "timeline__title";
      var text =
        (labels.lecture || "Lecture") +
        " " +
        page.num +
        " · " +
        (page.title[code] || page.title.en);

      if (page.file) {
        var link = document.createElement("a");
        link.href = page.file;
        link.textContent = text;
        title.appendChild(link);
      } else {
        title.appendChild(document.createTextNode(text));
        title.appendChild(span("timeline__badge", labels.planned || "Planned"));
      }
      item.appendChild(title);

      if (page.topics) {
        var topics = document.createElement("p");
        topics.className = "timeline__topics";
        topics.textContent = page.topics[code] || page.topics.en;
        item.appendChild(topics);
      }

      list.appendChild(item);
    });

    host.innerHTML = "";
    host.appendChild(list);
  }

  /* ------------------------------------------------------------------
     Sidebar drawer on small screens
     ------------------------------------------------------------------ */

  function initSidebarDrawer() {
    var toggle = $(".nav-toggle");
    var sidebar = $(".sidebar");
    var backdrop = $(".sidebar-backdrop");
    if (!toggle || !sidebar) return;

    function setOpen(open) {
      sidebar.classList.toggle("is-open", open);
      if (backdrop) backdrop.classList.toggle("is-visible", open);
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    }

    toggle.addEventListener("click", function () {
      setOpen(!sidebar.classList.contains("is-open"));
    });

    if (backdrop) {
      backdrop.addEventListener("click", function () {
        setOpen(false);
      });
    }

    // Jumping to a section should close the drawer behind it.
    sidebar.addEventListener("click", function (event) {
      if (event.target.closest("a")) setOpen(false);
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") setOpen(false);
    });
  }

  /* ------------------------------------------------------------------
     Code blocks: header bar with language label + copy button
     ------------------------------------------------------------------ */

  function initCodeBlocks() {
    var strings = t();

    $$(".code-block").forEach(function (block) {
      if ($(".code-block__head", block)) return;

      var code = $("code", block);
      if (!code) return;

      var head = document.createElement("div");
      head.className = "code-block__head";

      var label = span("code-block__lang", block.getAttribute("data-lang") || "text");

      var button = document.createElement("button");
      button.type = "button";
      button.className = "copy-btn";
      button.textContent = strings.copy;
      button.setAttribute("aria-label", strings.copy);

      button.addEventListener("click", function () {
        copyText(code.textContent).then(
          function () {
            flash(button, strings.copied);
          },
          function () {
            flash(button, strings.copyFailed);
          }
        );
      });

      head.appendChild(label);
      head.appendChild(button);
      block.insertBefore(head, block.firstChild);
    });
  }

  function flash(button, message) {
    var original = t().copy;
    button.textContent = message;
    button.classList.add("is-copied");
    window.setTimeout(function () {
      button.textContent = original;
      button.classList.remove("is-copied");
    }, 1600);
  }

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    // Fallback for file:// and older browsers.
    return new Promise(function (resolve, reject) {
      var area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      try {
        document.execCommand("copy") ? resolve() : reject();
      } catch (err) {
        reject(err);
      } finally {
        document.body.removeChild(area);
      }
    });
  }

  /* ------------------------------------------------------------------
     Regex playground
     ------------------------------------------------------------------ */

  function initRegexPlayground() {
    var root = $(".regex-playground");
    if (!root) return;

    var patternInput = $("[data-regex-pattern]", root);
    var flagsInput = $("[data-regex-flags]", root);
    var unicodeInput = $("[data-regex-unicode]", root);
    var textInput = $("[data-regex-text]", root);
    var output = $("[data-regex-output]", root);
    var status = $("[data-regex-status]", root);
    var list = $("[data-regex-list]", root);
    var runBtn = $("[data-regex-run]", root);

    if (!patternInput || !textInput || !output || !status) return;

    function buildFlags() {
      var flags = "g";
      if (flagsInput && flagsInput.checked) flags += "i";
      // The u flag enables \p{L} and friends — needed for Kazakh/Russian letters.
      if (unicodeInput && unicodeInput.checked) flags += "u";
      return flags;
    }

    function run() {
      var pattern = patternInput.value;
      var text = textInput.value;

      status.classList.remove("is-error");
      if (list) list.innerHTML = "";

      if (!pattern) {
        output.textContent = text;
        status.textContent = t().emptyPattern;
        return;
      }

      var re;
      try {
        re = new RegExp(pattern, buildFlags());
      } catch (err) {
        output.textContent = text;
        status.textContent = t().invalid;
        status.classList.add("is-error");
        return;
      }

      var matches = [];
      var html = "";
      var lastIndex = 0;
      var guard = 0;
      var match;

      while ((match = re.exec(text)) !== null && guard < 5000) {
        guard += 1;
        // Zero-length matches would loop forever — step past them.
        if (match.index === re.lastIndex) {
          re.lastIndex += 1;
          continue;
        }
        matches.push(match[0]);
        html += escapeHtml(text.slice(lastIndex, match.index));
        html += '<mark class="match">' + escapeHtml(match[0]) + "</mark>";
        lastIndex = match.index + match[0].length;
      }

      html += escapeHtml(text.slice(lastIndex));
      output.innerHTML = html;
      status.textContent = matches.length
        ? t().matches(matches.length)
        : t().noMatches;

      if (list) {
        matches.forEach(function (value) {
          var item = document.createElement("li");
          item.textContent = value;
          list.appendChild(item);
        });
      }
    }

    patternInput.addEventListener("input", run);
    textInput.addEventListener("input", run);
    if (flagsInput) flagsInput.addEventListener("change", run);
    if (unicodeInput) unicodeInput.addEventListener("change", run);
    if (runBtn) runBtn.addEventListener("click", run);

    $$("[data-preset-pattern]", root).forEach(function (button) {
      button.addEventListener("click", function () {
        patternInput.value = button.getAttribute("data-preset-pattern") || "";
        var presetText = button.getAttribute("data-preset-text");
        if (presetText) textInput.value = presetText;
        if (unicodeInput) {
          unicodeInput.checked = button.getAttribute("data-preset-unicode") !== null;
        }
        run();
        patternInput.focus();
      });
    });

    run();
  }

  function escapeHtml(value) {
    return value
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  /* ------------------------------------------------------------------
     Quiz — self-check questions with immediate feedback
     ------------------------------------------------------------------ */

  function initQuizzes() {
    $$("[data-quiz]").forEach(setUpQuiz);
  }

  function setUpQuiz(quiz) {
    var strings = t();
    var items = $$(".quiz__item", quiz);
    if (!items.length) return;

    var score = document.createElement("p");
    score.className = "quiz__score";
    score.setAttribute("role", "status");

    var reset = document.createElement("button");
    reset.type = "button";
    reset.className = "btn";
    reset.textContent = strings.reset;

    var actions = document.createElement("div");
    actions.className = "quiz__actions";
    actions.appendChild(score);
    actions.appendChild(reset);
    quiz.parentNode.insertBefore(actions, quiz.nextSibling);

    var answered = 0;
    var correct = 0;

    function paintScore() {
      score.textContent = strings.score(correct, answered, items.length);
      score.classList.toggle("is-complete", answered === items.length);
    }

    items.forEach(function (item) {
      var options = $$(".quiz__option", item);
      var feedback = $(".quiz__feedback", item);

      options.forEach(function (option, index) {
        option.setAttribute("data-key", String.fromCharCode(65 + index));

        option.addEventListener("click", function () {
          if (item.getAttribute("data-answered")) return;
          item.setAttribute("data-answered", "true");
          answered += 1;

          var isCorrect = option.getAttribute("data-correct") !== null;
          if (isCorrect) correct += 1;

          options.forEach(function (other) {
            other.disabled = true;
            if (other.getAttribute("data-correct") !== null) {
              other.classList.add("is-correct");
            }
          });
          if (!isCorrect) option.classList.add("is-wrong");

          if (feedback) feedback.hidden = false;
          paintScore();
        });
      });
    });

    reset.addEventListener("click", function () {
      answered = 0;
      correct = 0;
      items.forEach(function (item) {
        item.removeAttribute("data-answered");
        var feedback = $(".quiz__feedback", item);
        if (feedback) feedback.hidden = true;
        $$(".quiz__option", item).forEach(function (option) {
          option.disabled = false;
          option.classList.remove("is-correct");
          option.classList.remove("is-wrong");
        });
      });
      paintScore();
    });

    paintScore();
  }

  /* ------------------------------------------------------------------
     Reading progress indicator
     ------------------------------------------------------------------ */

  function initProgressBar() {
    var bar = $(".progress-bar");
    if (!bar) return;

    var ticking = false;

    function update() {
      var height = document.documentElement.scrollHeight - window.innerHeight;
      var ratio = height > 0 ? window.scrollY / height : 0;
      bar.style.transform = "scaleX(" + Math.min(Math.max(ratio, 0), 1) + ")";
      ticking = false;
    }

    window.addEventListener(
      "scroll",
      function () {
        if (!ticking) {
          window.requestAnimationFrame(update);
          ticking = true;
        }
      },
      { passive: true }
    );

    window.addEventListener("resize", update);
    update();
  }

  /* ------------------------------------------------------------------
     Language preference (used by the root landing page)
     ------------------------------------------------------------------ */

  var LANG_KEY = "nlp-course-lang";

  function initLanguageMemory() {
    $$("[data-lang-link]").forEach(function (link) {
      link.addEventListener("click", function () {
        try {
          window.localStorage.setItem(
            LANG_KEY,
            link.getAttribute("data-lang-link")
          );
        } catch (err) {
          /* storage may be unavailable — the link still works */
        }
      });
    });

    var hint = $("[data-lang-hint]");
    if (!hint) return;

    var saved = null;
    try {
      saved = window.localStorage.getItem(LANG_KEY);
    } catch (err) {
      saved = null;
    }
    if (!saved) return;

    var target = $('[data-lang-link="' + saved + '"]');
    if (target) {
      target.classList.add("btn--primary");
      hint.hidden = false;
    }
  }

  /* ------------------------------------------------------------------
     Boot
     ------------------------------------------------------------------ */

  function init() {
    initRoadmap();
    initTableOfContents();
    initSidebarDrawer();
    initCodeBlocks();
    initRegexPlayground();
    initQuizzes();
    initProgressBar();
    initLanguageMemory();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
