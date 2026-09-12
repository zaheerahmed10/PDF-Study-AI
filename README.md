# 📚 PDF Study AI

An AI-powered document study assistant built with **Streamlit**, **FAISS**, **Sentence Transformers**, and **Groq**. Upload any PDF, ask questions about it, and get clear, source-backed answers — with page-level citations and exam-ready answer modes.

---

## ✨ Features

- **📖 PDF Based Answers** — Answers are generated strictly from the content of your uploaded PDF.
- **🔎 Semantic Search (RAG)** — Uses vector embeddings + FAISS to find the most relevant chunks of text for your question.
- **📑 Page Sources** — Every answer shows which PDF pages were used to generate it.
- **🎓 Exam Ready Modes** — Choose from Short, Medium, Detailed, or exam-style answers (2 / 5 / 10 Marks).
- **🎨 Clean, Modern UI** — Premium SaaS-style interface built with custom CSS on top of Streamlit.

---

## 🛠️ Tech Stack

| Component            | Technology                                  |
|-----------------------|----------------------------------------------|
| Frontend / App        | [Streamlit](https://streamlit.io)            |
| PDF Text Extraction   | [pypdf](https://pypi.org/project/pypdf/)      |
| Embeddings            | [Sentence Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| Vector Search          | [FAISS](https://github.com/facebookresearch/faiss) |
| LLM / Answer Generation | [Groq API](https://groq.com/) (`openai/gpt-oss-20b`) |

---

## ⚙️ How It Works

1. **Upload a PDF** — the app extracts text page by page using `pypdf`.
2. **Chunking** — extracted text is split into overlapping chunks (900 chars, 150 overlap) for better retrieval accuracy.
3. **Embedding & Indexing** — each chunk is embedded with `all-MiniLM-L6-v2` and stored in a FAISS vector index (`IndexFlatIP`).
4. **Question → Retrieval** — your question is embedded and the top 5 most relevant chunks are retrieved via cosine similarity search.
5. **Answer Generation** — retrieved chunks are passed as context to Groq's LLM, which generates an answer **strictly based on the PDF content** (no outside knowledge, no hallucination).
6. **Sources Displayed** — the pages used to generate the answer are shown below it for transparency.

---

## 🚀 Getting Started (Local Setup)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key

Create a `.streamlit/secrets.toml` file in the project root:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Get a free API key from [console.groq.com](https://console.groq.com).

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Push this project to a GitHub repository (must include `app.py` and `requirements.txt`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select your repo/branch, and set the main file path to `app.py`.
4. Under **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
5. Deploy — the first build may take a few minutes since it installs `torch` and `sentence-transformers`.

---

## 📁 Project Structure

```
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── secrets.toml        # API keys (not committed to git)
└── README.md
```

---

## 📦 requirements.txt

```
streamlit
pypdf
sentence-transformers
faiss-cpu
groq
torch
numpy
```

---

## ⚠️ Notes

- Only text-based PDFs are supported — scanned/image-only PDFs with no extractable text will show an error.
- Answers are constrained to the uploaded PDF's content; if the answer isn't found, the app explicitly states that instead of guessing.
- Do not commit your `secrets.toml` file or expose your Groq API key publicly.

---
