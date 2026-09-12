import streamlit as st
import hashlib
import tempfile
import os
import re
import textwrap

import numpy as np
import faiss
import torch

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PDF Study AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# HTML RENDER HELPER (fixes markdown code-block issue)
# =========================================================

def render_html(html):
    """Strips leading indentation so Streamlit's markdown parser
    doesn't mistake indented HTML for a code block."""
    st.markdown(textwrap.dedent(html), unsafe_allow_html=True)


# =========================================================
# CUSTOM CSS
# =========================================================

render_html("""
<style>

    /* ---------- Global ---------- */

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: #fdf6e8;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    /* ---------- Navbar ---------- */

    .navbar {
        background: #ffffff;
        border-radius: 14px;
        padding: 16px 28px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    .brand { display: flex; align-items: center; gap: 12px; }

    .brand-icon {
        width: 42px;
        height: 42px;
        border-radius: 10px;
        background: #fff1e0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }

    .brand-title {
        font-size: 22px;
        font-weight: 800;
        color: #f5821f;
        letter-spacing: -0.3px;
        line-height: 1.2;
    }

    .brand-subtitle {
        font-size: 12px;
        color: #8b8378;
        margin-top: 1px;
    }

    .nav-badge {
        background: #f5821f;
        color: #ffffff;
        padding: 9px 20px;
        border-radius: 10px;
        font-size: 13px;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .nav-badge .dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #ffffff;
        display: inline-block;
    }

    /* ---------- Hero ---------- */

    .hero {
        background: #ffffff;
        border-radius: 20px;
        padding: 46px 30px;
        margin-bottom: 24px;
        text-align: center;
        box-shadow: 0 2px 14px rgba(0,0,0,0.05);
    }

    .hero-title {
        font-size: 44px;
        font-weight: 800;
        color: #f5821f;
        letter-spacing: -1px;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        font-size: 20px;
        font-weight: 700;
        color: #2a2a2a;
        margin-bottom: 14px;
    }

    .hero-text {
        color: #8b8378;
        font-size: 15px;
        line-height: 1.6;
        max-width: 600px;
        margin: 0 auto;
    }

    /* ---------- Cards ---------- */

    .info-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 18px 20px;
        height: 100%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }

    .info-label {
        color: #b3a99a;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 8px;
    }

    .info-value {
        color: #2a2a2a;
        font-size: 15px;
        font-weight: 700;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: #2fa84f;
        font-size: 14px;
        font-weight: 700;
    }

    .status-pill .dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #2fa84f;
    }

    /* ---------- Section titles ---------- */

    .section-title {
        font-size: 19px;
        font-weight: 800;
        color: #2a2a2a;
        margin-top: 28px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .section-hint {
        font-size: 13px;
        color: #8b8378;
        margin-top: -6px;
        margin-bottom: 14px;
    }

    /* ---------- Answer card ---------- */

    .answer-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 4px 28px 24px 28px;
        margin-top: 8px;
        box-shadow: 0 2px 14px rgba(0,0,0,0.05);
    }

    .answer-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 18px 0 14px 0;
        border-bottom: 2px solid #fdf6e8;
        margin-bottom: 16px;
    }

    .answer-label {
        font-size: 13px;
        font-weight: 800;
        color: #f5821f;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .answer-mode-badge {
        background: #fff1e0;
        color: #f5821f;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }

    .answer-card .stMarkdown p {
        color: #2a2a2a;
        font-size: 15px;
        line-height: 1.75;
    }

    /* ---------- Sources ---------- */

    .source-card {
        background: #ffffff;
        border-left: 4px solid #f5821f;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .source-page {
        color: #f5821f;
        font-weight: 800;
        font-size: 13.5px;
        margin-bottom: 4px;
    }

    .source-text {
        color: #8b8378;
        font-size: 12.5px;
        line-height: 1.55;
    }

    /* ---------- Footer ---------- */

    .custom-footer {
        text-align: center;
        color: #b3a99a;
        font-size: 12px;
        padding: 26px 0 8px 0;
        margin-top: 40px;
        border-top: 2px solid #f0e6d2;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: #ffffff;
    }

    section[data-testid="stSidebar"] .block-container {
        padding: 24px 18px;
    }

    .sidebar-heading {
        font-size: 11px;
        font-weight: 800;
        color: #f5821f;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 4px;
    }

    .sidebar-caption {
        font-size: 12.5px;
        color: #8b8378;
        margin-bottom: 12px;
    }

    .feature-item {
        display: flex;
        gap: 10px;
        padding: 10px 0;
        border-bottom: 1px solid #f4ede0;
    }

    .feature-item:last-child { border-bottom: none; }

    .feature-icon { font-size: 15px; margin-top: 1px; }

    .feature-title {
        font-size: 12.5px;
        font-weight: 700;
        color: #2a2a2a;
    }

    .feature-desc {
        font-size: 11.5px;
        color: #b3a99a;
        line-height: 1.4;
        margin-top: 1px;
    }

    /* ---------- Streamlit widget tweaks ---------- */

    .stButton > button {
        border-radius: 12px;
        font-weight: 800;
        font-size: 16px;
        min-height: 52px;
        background: #f5821f;
        color: #ffffff;
        border: none;
        transition: background 0.15s ease;
    }

    .stButton > button:hover {
        background: #d96e12;
        color: #ffffff;
    }

    .stButton > button:disabled {
        background: #f0e6d2;
        color: #c8bda8;
    }

    .stTextArea textarea {
        border-radius: 14px;
        border: 2px solid #f0e6d2;
        font-size: 14.5px;
        padding: 16px;
        background: #fffdf9;
    }

    .stTextArea textarea:focus {
        border-color: #f5821f;
        box-shadow: 0 0 0 3px rgba(245, 130, 31, 0.15);
    }

    div[data-testid="stFileUploader"] {
        border: 2px dashed #f5821f;
        border-radius: 14px;
        padding: 12px;
        background: #fff8ee;
    }

    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #f0e6d2;
    }

    div[data-testid="stStatusWidget"] {
        border-radius: 14px;
    }

</style>
""")


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "pdf_name": None,
    "pdf_hash": None,
    "pages": [],
    "chunks": [],
    "index": None,
    "answer": None,
    "sources": [],
    "processing": False,
    "question": ""
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# =========================================================
# GROQ
# =========================================================

try:

    groq_client = Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )

except Exception:

    groq_client = None


# =========================================================
# NAVBAR
# =========================================================

render_html("""
<div class="navbar">
<div class="brand">
<div class="brand-icon">📚</div>
<div>
<div class="brand-title">PDF Study AI</div>
<div class="brand-subtitle">AI-powered document study assistant</div>
</div>
</div>
<div class="nav-badge">
<span class="dot"></span> RAG Active
</div>
</div>
""")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown('<div class="sidebar-heading">Document</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-caption">Upload a PDF and ask questions from it.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown('<div class="sidebar-heading">Answer Mode</div>', unsafe_allow_html=True)

    answer_mode = st.selectbox(
        "Choose answer length",
        [
            "Short",
            "Medium",
            "Detailed",
            "2 Marks",
            "5 Marks",
            "10 Marks"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown('<div class="sidebar-heading">Features</div>', unsafe_allow_html=True)

    render_html("""
    <div class="feature-item">
    <div class="feature-icon">📖</div>
    <div>
    <div class="feature-title">PDF Based Answers</div>
    <div class="feature-desc">Answers are generated from your uploaded PDF.</div>
    </div>
    </div>

    <div class="feature-item">
    <div class="feature-icon">🔎</div>
    <div>
    <div class="feature-title">Semantic Search</div>
    <div class="feature-desc">Finds the most relevant content.</div>
    </div>
    </div>

    <div class="feature-item">
    <div class="feature-icon">📑</div>
    <div>
    <div class="feature-title">Page Sources</div>
    <div class="feature-desc">See which pages were used.</div>
    </div>
    </div>

    <div class="feature-item">
    <div class="feature-icon">🎓</div>
    <div>
    <div class="feature-title">Exam Ready</div>
    <div class="feature-desc">Get answers for 2, 5 and 10 marks.</div>
    </div>
    </div>
    """)

    st.divider()

    st.caption("PDF Study AI • Portfolio Project")


# =========================================================
# FUNCTIONS
# =========================================================

def clean_text(text):

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_pages(pdf_bytes):

    pages = []

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp:

        temp.write(pdf_bytes)
        temp_path = temp.name

    try:

        reader = PdfReader(temp_path)

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text() or ""

            text = clean_text(text)

            if text:

                pages.append({
                    "page": page_number,
                    "text": text
                })

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)

    return pages


def create_chunks(
    pages,
    chunk_size=900,
    overlap=150
):

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]
        text = page_data["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            if chunk_text.strip():

                chunks.append({
                    "page": page_number,
                    "text": chunk_text
                })

            start += chunk_size - overlap

    return chunks


def create_vector_index(chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


def retrieve_chunks(
    question,
    chunks,
    index,
    top_k=5
):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = question_embedding.astype(
        "float32"
    )

    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        result = chunks[idx].copy()

        result["score"] = float(score)

        results.append(result)

    return results


def get_answer_instruction(mode):

    instructions = {

        "Short":
            "Answer in 2-4 clear sentences.",

        "Medium":
            "Answer in a clear paragraph with the important points.",

        "Detailed":
            "Give a detailed explanation using headings or bullet points where useful.",

        "2 Marks":
            "Give a concise exam-style answer suitable for 2 marks.",

        "5 Marks":
            "Give a structured exam-style answer suitable for 5 marks.",

        "10 Marks":
            "Give a detailed, well-structured exam-style answer suitable for 10 marks."
    }

    return instructions[mode]


def generate_answer(
    question,
    retrieved_chunks,
    mode
):

    if groq_client is None:

        return "Groq API key is not configured."

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"[Page {chunk['page']}]\n{chunk['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    instruction = get_answer_instruction(
        mode
    )

    prompt = f"""
You are PDF Study AI, an academic assistant.

Answer the user's question ONLY using the provided PDF context.

Important rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not available in the PDF, clearly say:
   "This information is not available in the uploaded PDF."
4. Keep the answer easy to understand.
5. Use clean formatting.
6. {instruction}

PDF CONTEXT:

{context}

USER QUESTION:

{question}
"""

    response = groq_client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "system",
                "content": "You are a helpful PDF study assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.15,

        max_tokens=1000,

        include_reasoning=False
    )

    return response.choices[0].message.content


def make_hash(data):

    return hashlib.md5(
        data
    ).hexdigest()


# =========================================================
# HERO
# =========================================================

render_html("""
<div class="hero">
<div class="hero-title">PDF Study AI</div>
<div class="hero-subtitle">using RAG + Groq LLM</div>
<div class="hero-text">
Upload any PDF, ask questions, and get clear, source-backed answers based on the content of your document.
</div>
</div>
""")


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    current_hash = make_hash(
        pdf_bytes
    )

    if current_hash != st.session_state.pdf_hash:

        st.session_state.pdf_name = uploaded_file.name
        st.session_state.pdf_hash = current_hash
        st.session_state.answer = None
        st.session_state.sources = []

        with st.status(
            "Processing your PDF...",
            expanded=True
        ) as status:

            st.write("📖 Reading PDF pages...")

            pages = extract_pages(
                pdf_bytes
            )

            if not pages:

                st.error(
                    "No readable text was found in this PDF."
                )

                st.stop()

            st.session_state.pages = pages

            st.write(
                f"✅ {len(pages)} pages extracted."
            )

            st.write(
                "✂️ Creating text chunks..."
            )

            chunks = create_chunks(
                pages
            )

            st.session_state.chunks = chunks

            st.write(
                f"✅ {len(chunks)} chunks created."
            )

            st.write(
                "🔎 Building semantic search index..."
            )

            index = create_vector_index(
                chunks
            )

            st.session_state.index = index

            st.write(
                "✅ Document is ready."
            )

            status.update(
                label="PDF ready!",
                state="complete"
            )


# =========================================================
# DOCUMENT INFO
# =========================================================

if st.session_state.pdf_name:

    col1, col2, col3 = st.columns(3)

    with col1:

        render_html(
            f"""
            <div class="info-card">
            <div class="info-label">Document</div>
            <div class="info-value">📄 {st.session_state.pdf_name}</div>
            </div>
            """
        )

    with col2:

        render_html(
            f"""
            <div class="info-card">
            <div class="info-label">Pages</div>
            <div class="info-value">📖 {len(st.session_state.pages)}</div>
            </div>
            """
        )

    with col3:

        render_html(
            """
            <div class="info-card">
            <div class="info-label">Status</div>
            <div class="status-pill"><span class="dot"></span> Ready</div>
            </div>
            """
        )


# =========================================================
# QUESTION AREA
# =========================================================

render_html("""
<div class="section-title">💬 Ask your PDF</div>
<div class="section-hint">Type a question about the uploaded document to get a source-backed answer.</div>
""")


question = st.text_area(
    "Question",
    placeholder="Enter your question here...",
    height=120,
    label_visibility="collapsed"
)


# =========================================================
# ASK BUTTON
# =========================================================

ask_button = st.button(
    "Ask Question",
    type="primary",
    use_container_width=True,
    disabled=(
        not question.strip()
        or st.session_state.index is None
    )
)


# =========================================================
# ANSWER
# =========================================================

if ask_button:

    with st.status(
        "Finding the best answer...",
        expanded=True
    ) as status:

        st.write(
            "🔎 Searching relevant pages..."
        )

        retrieved = retrieve_chunks(
            question,
            st.session_state.chunks,
            st.session_state.index,
            top_k=5
        )

        st.write(
            "📚 Preparing PDF context..."
        )

        st.write(
            "🤖 Generating answer..."
        )

        try:

            answer = generate_answer(
                question,
                retrieved,
                answer_mode
            )

            st.session_state.answer = answer

            st.session_state.sources = retrieved

            status.update(
                label="Answer ready!",
                state="complete"
            )

        except Exception as e:

            status.update(
                label="Something went wrong.",
                state="error"
            )

            st.error(
                f"Error: {str(e)}"
            )


# =========================================================
# DISPLAY ANSWER
# =========================================================

if st.session_state.answer:

    render_html('<div class="section-title">📝 Answer</div>')

    render_html('<div class="answer-card">')

    render_html(
        f"""
        <div class="answer-header">
        <div class="answer-label">🤖 AI Answer</div>
        <div class="answer-mode-badge">{answer_mode}</div>
        </div>
        """
    )

    # Native Markdown rendering so headings, bullets, bold, code etc. render properly
    st.markdown(st.session_state.answer)

    render_html('</div>')


# =========================================================
# SOURCES
# =========================================================

if st.session_state.sources:

    render_html('<div class="section-title">📑 Sources from PDF</div>')

    shown_pages = set()

    for source in st.session_state.sources:

        page = source["page"]

        if page in shown_pages:
            continue

        shown_pages.add(page)

        preview = source["text"][:250]

        render_html(
            f"""
            <div class="source-card">
            <div class="source-page">📄 Page {page}</div>
            <div class="source-text">{preview}...</div>
            </div>
            """
        )


# =========================================================
# FOOTER
# =========================================================

render_html("""
<div class="custom-footer">
📚 <b>PDF Study AI</b> — Built with Streamlit, FAISS, Sentence Transformers & Groq
</div>
""")