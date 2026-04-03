"""
Umberlith MUN Resolution Architect
====================================
Optimised version:
  - spaCy model loaded once via @st.cache_resource
  - Clause-level analysis (preambulatory vs operative)
  - Expanded forbidden-verb and starter-phrase dictionaries
  - Passive-voice detection with suggested rewrites
  - Named-entity actor detection
  - Readability / density metrics
  - Download button for enhanced draft
  - Clean, structured UI with expanders and tabs
"""

import streamlit as st
import subprocess, sys, re

try:
    import spacy
    spacy.load("en_core_web_sm")
except (ImportError, OSError):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
    subprocess.check_call([
        sys.executable,  "-m", "pip", "install",
        "https://github.com/explosion/spacy-models/releases/download/"
        "en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
    ])
    import spacy

import re

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="MUN Resolution Architect",
    page_icon="🇺🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load NLP model once ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading NLP engine…")
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

# ── Data ───────────────────────────────────────────────────────────────────
FORBIDDEN_LEMMAS: dict[str, str] = {
    "think":    "believes",
    "want":     "urges",
    "ask":      "requests",
    "say":      "declares",
    "help":     "facilitates",
    "need":     "necessitates",
    "try":      "endeavors",
    "tell":     "reminds",
    "show":     "demonstrates",
    "use":      "utilizes",
    "make":     "establishes",
    "get":      "obtains",
    "give":     "provides",
    "look":     "examines",
    "know":     "recognizes",
    "hope":     "aspires",
}

# Standard MUN clause openers (lowercase lemma → canonical form)
PREAMBULATORY_OPENERS = {
    "acknowledge", "affirm", "alarmed", "aware", "bearing", "believing",
    "concerned", "confident", "convinced", "declaring", "deeply",
    "deploring", "desiring", "emphasizing", "expecting", "expressing",
    "fulfilling", "further", "guided", "having", "keeping", "mindful",
    "noting", "observing", "reaffirming", "realizing", "recalling",
    "recognizing", "referring", "regretting", "seeking", "stressing",
    "taking", "underlining", "viewing", "welcoming",
}

OPERATIVE_OPENERS = {
    "accepts", "affirms", "approves", "authorizes", "calls", "commends",
    "condemns", "confirms", "congratulates", "considers", "decides",
    "declares", "demands", "deplores", "directs", "draws", "emphasizes",
    "encourages", "endorses", "expresses", "further", "invites",
    "notes", "proclaims", "reaffirms", "recommends", "regrets",
    "reminds", "requests", "resolves", "stresses", "strongly",
    "supports", "takes", "transmits", "trusts", "urges", "welcomes",
}

POS_LABELS = {
    "NOUN": "Nouns", "VERB": "Verbs", "ADJ": "Adjectives",
    "ADV": "Adverbs", "PROPN": "Proper nouns", "ADP": "Prepositions",
    "CONJ": "Conjunctions", "CCONJ": "Conjunctions", "PRON": "Pronouns",
    "DET": "Determiners", "AUX": "Auxiliaries", "NUM": "Numbers",
    "PUNCT": "Punctuation", "SPACE": "Spaces", "SYM": "Symbols",
    "X": "Other", "INTJ": "Interjections", "PART": "Particles",
    "SCONJ": "Sub-conjunctions",
}

# ── Analysis engine ────────────────────────────────────────────────────────
class ResolutionEngine:
    def __init__(self, text: str):
        self.raw = text
        self.doc = nlp(text)
        self.issues: list[dict] = []   # {sentence, type, original, suggestion}

    # ------------------------------------------------------------------
    def analyze(self) -> str:
        """Run all checks; return improved text."""
        improved_sentences = []
        for sent in self.doc.sents:
            improved_tokens = []
            for token in sent:
                word = token.text_with_ws  # preserves spacing

                # 1. Weak-verb substitution
                if token.pos_ == "VERB" and token.lemma_ in FORBIDDEN_LEMMAS:
                    replacement = FORBIDDEN_LEMMAS[token.lemma_]
                    if token.text.istitle():
                        replacement = replacement.title()
                    self.issues.append({
                        "sentence": sent.text.strip(),
                        "type": "⚠️ Weak Verb",
                        "original": token.text,
                        "suggestion": f"Replace **'{token.text}'** with **'{replacement}'** for diplomatic weight.",
                    })
                    word = replacement + token.whitespace_

                # 2. Passive voice
                if token.dep_ == "auxpass":
                    self.issues.append({
                        "sentence": sent.text.strip(),
                        "type": "📝 Passive Voice",
                        "original": sent.text.strip(),
                        "suggestion": "Rewrite in active voice — e.g. *'The Council decides…'* instead of *'It is decided…'*",
                    })

                improved_tokens.append(word)

            improved_sentences.append("".join(improved_tokens))

        return "".join(improved_sentences)

    # ------------------------------------------------------------------
    def classify_clauses(self) -> list[dict]:
        """Tag each sentence as Preambulatory, Operative, or Unknown."""
        clauses = []
        for sent in self.doc.sents:
            text = sent.text.strip()
            if not text:
                continue
            first_word = text.split()[0].lower().rstrip(",") if text.split() else ""
            if first_word in PREAMBULATORY_OPENERS:
                kind = "Preambulatory"
                colour = "🟡"
            elif first_word in OPERATIVE_OPENERS:
                kind = "Operative"
                colour = "🟢"
            else:
                kind = "Unclassified"
                colour = "⚪"
            clauses.append({"text": text, "kind": kind, "colour": colour})
        return clauses

    # ------------------------------------------------------------------
    def find_actors(self) -> list[str]:
        """Named-entity countries and organisations."""
        return sorted({
            ent.text for ent in self.doc.ents
            if ent.label_ in ("GPE", "ORG", "NORP")
        })

    # ------------------------------------------------------------------
    def pos_stats(self) -> dict[str, int]:
        """Human-readable POS frequency map."""
        raw = self.doc.count_by(spacy.attrs.POS)
        return {
            POS_LABELS.get(self.doc.vocab.strings[k], self.doc.vocab.strings[k]): v
            for k, v in raw.items()
            if self.doc.vocab.strings[k] not in ("PUNCT", "SPACE", "X", "SYM")
        }

    # ------------------------------------------------------------------
    def readability_metrics(self) -> dict:
        sentences = list(self.doc.sents)
        tokens = [t for t in self.doc if not t.is_punct and not t.is_space]
        words = [t for t in tokens if t.is_alpha]
        avg_sent = round(len(words) / max(len(sentences), 1), 1)
        avg_word = round(sum(len(w.text) for w in words) / max(len(words), 1), 1)
        return {
            "Sentences": len(sentences),
            "Words": len(words),
            "Avg words / sentence": avg_sent,
            "Avg word length": avg_word,
            "Unique lemmas": len({t.lemma_ for t in words}),
        }


# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar branding */
    section[data-testid="stSidebar"] { background: #0a1628; }
    section[data-testid="stSidebar"] * { color: #e8edf5 !important; }
    section[data-testid="stSidebar"] .stMarkdown h2 { color: #7eb8f7 !important; }

    /* Issue cards */
    .issue-card {
        background: #1a2744;
        border-left: 4px solid #f0a500;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 10px;
        font-size: 0.9rem;
    }
    .issue-card .issue-type { font-weight: 700; margin-bottom: 4px; }
    .issue-card .issue-suggestion { color: #a8c7fa; margin-top: 4px; }

    /* Clause tags */
    .clause-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .tag-preambulatory { background: #3d3000; color: #f0c040; }
    .tag-operative     { background: #003020; color: #40d080; }
    .tag-unclassified  { background: #2a2a2a; color: #aaaaaa; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🇺🇳 MUN Architect")
    st.markdown("**Resolution Architect Pro**")
    st.divider()

    st.markdown("### ⚙️ Analysis options")
    check_weak_verbs   = st.toggle("Detect weak verbs",   value=True)
    check_passive      = st.toggle("Detect passive voice", value=True)
    check_clauses      = st.toggle("Classify clauses",     value=True)
    check_actors       = st.toggle("Detect actors (NER)",  value=True)

    st.divider()
    st.markdown("### 📖 Quick reference")
    with st.expander("Preambulatory openers"):
        st.write(", ".join(sorted(PREAMBULATORY_OPENERS)))
    with st.expander("Operative openers"):
        st.write(", ".join(sorted(OPERATIVE_OPENERS)))
    with st.expander("Weak → strong verbs"):
        for weak, strong in sorted(FORBIDDEN_LEMMAS.items()):
            st.markdown(f"- *{weak}* → **{strong}**")

    st.divider()
    st.caption("Powered by spaCy `en_core_web_sm`")


# ── Main area ──────────────────────────────────────────────────────────────
st.title("Resolution Architect Pro")
st.caption("NLP (NAtural Language Processing) -powered drafting assistant for Model UN resolutions")

raw_input = st.text_area(
    "Paste your draft resolution or a single clause:",
    height=220,
    placeholder=(
        "Put your content here!"
    ),
)

run = st.button("🚀 Analyse Resolution", type="primary", use_container_width=True)

if run:
    if not raw_input.strip():
        st.warning("Please enter some text before running the analysis.")
        st.stop()

    engine = ResolutionEngine(raw_input)
    improved = engine.analyze()
    clauses  = engine.classify_clauses() if check_clauses  else []
    actors   = engine.find_actors()      if check_actors   else []
    metrics  = engine.readability_metrics()

    # Filter issues based on toggles
    visible_issues = [
        i for i in engine.issues
        if (i["type"].startswith("⚠️") and check_weak_verbs)
        or (i["type"].startswith("📝") and check_passive)
    ]

    # ── Metric strip ──────────────────────────────────────────────────
    m_cols = st.columns(len(metrics))
    for col, (label, val) in zip(m_cols, metrics.items()):
        col.metric(label, val)

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Issues & fixes",
        "✨ Enhanced draft",
        "📄 Clause map",
        "📊 Linguistic stats",
    ])

    # ── Tab 1: Issues ─────────────────────────────────────────────────
    with tab1:
        if not visible_issues:
            st.success("✅ No issues detected with the selected checks.")
        else:
            st.markdown(f"**{len(visible_issues)} issue(s) found:**")
            for issue in visible_issues:
                st.markdown(f"""
<div class="issue-card">
  <div class="issue-type">{issue['type']}</div>
  <div><em>"{issue['original']}"</em></div>
  <div class="issue-suggestion">💡 {issue['suggestion']}</div>
</div>
""", unsafe_allow_html=True)

        if actors:
            st.divider()
            st.markdown("**🌍 Detected actors (countries / organisations):**")
            st.markdown(" · ".join(f"`{a}`" for a in actors))

    # ── Tab 2: Enhanced draft ─────────────────────────────────────────
    with tab2:
        st.markdown("#### Suggested revision")
        st.info(improved)
        st.download_button(
            label="⬇️ Download enhanced draft (.txt)",
            data=improved,
            file_name="enhanced_resolution.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # ── Tab 3: Clause map ─────────────────────────────────────────────
    with tab3:
        if not check_clauses:
            st.info("Enable 'Classify clauses' in the sidebar to use this tab.")
        elif not clauses:
            st.warning("No clauses detected. Try separating clauses with commas or new lines.")
        else:
            preamb = sum(1 for c in clauses if c["kind"] == "Preambulatory")
            operat = sum(1 for c in clauses if c["kind"] == "Operative")
            unclsf = sum(1 for c in clauses if c["kind"] == "Unclassified")

            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Preambulatory", preamb)
            cc2.metric("Operative", operat)
            cc3.metric("Unclassified", unclsf)

            st.divider()
            for clause in clauses:
                tag_class = f"tag-{clause['kind'].lower()}"
                st.markdown(
                    f"<span class='clause-tag {tag_class}'>{clause['colour']} {clause['kind']}</span> {clause['text']}",
                    unsafe_allow_html=True,
                )
                st.write("")  # spacing

    # ── Tab 4: Stats ──────────────────────────────────────────────────
    with tab4:
        pos_data = engine.pos_stats()
        if pos_data:
            st.markdown("#### Parts-of-speech distribution")
            st.bar_chart(pos_data)
        else:
            st.info("Not enough tokens to generate statistics.")

elif not run:
    st.markdown("""
---
**How to use:**
1. Paste your draft resolution (full document or individual clauses) in the text box above.
2. Toggle analysis options in the sidebar.
3. Click **Analyse Resolution** to get instant feedback.

**Tabs explained:**
| Tab | What you get |
|-----|-------------|
| 🔍 Issues & fixes | Weak verbs, passive voice, suggested replacements |
| ✨ Enhanced draft | Rewritten text with substitutions applied |
| 📄 Clause map | Each sentence tagged as Preambulatory / Operative |
| 📊 Linguistic stats | Parts-of-speech chart for your text |
""")


col = st.columns(7)[3]
click = col.link_button("GitHub", "https://github.com/adamsaou")

st.text("-This App is still under developpement")
