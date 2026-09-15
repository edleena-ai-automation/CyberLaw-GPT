import io
import hashlib
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Pakistan Cyber Law AI Assistant",
    page_icon="⚖️",
    layout="wide",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.25rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .law-box {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,.25);
        background: rgba(128,128,128,.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">⚖️ Pakistan Cyber Law AI Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">RAG-based assistant grounded in your uploaded Pakistan cyber-law PDF • Text + Voice Input</div>',
    unsafe_allow_html=True,
)

# -----------------------------
# Configuration
# -----------------------------
LLM_MODEL = "openai/gpt-oss-120b"
STT_MODEL = "whisper-large-v3-turbo"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# -----------------------------
# Secrets / API
# -----------------------------
api_key = st.secrets.get("GROQ_API_KEY", "")

if not api_key:
    st.error(
        "GROQ_API_KEY is not configured. Add it in Streamlit Cloud → "
        "App settings → Secrets."
    )
    st.code('GROQ_API_KEY = "gsk_..."')
    st.stop()

client = Groq(api_key=api_key)

# -----------------------------
# Cached embedding model
# -----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


embedder = load_embedding_model()

# -----------------------------
# PDF helpers
# -----------------------------
def extract_pdf_text(pdf_bytes: bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []

    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text:
            pages.append((page_no, text))

    return pages


def chunk_pages(pages, chunk_size=1100, overlap=180):
    chunks = []

    for page_no, text in pages:
        if len(text) <= chunk_size:
            chunks.append(
                {
                    "text": text,
                    "page": page_no,
                }
            )
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(
                    {
                        "text": chunk,
                        "page": page_no,
                    }
                )

            if end >= len(text):
                break

            start = max(0, end - overlap)

    return chunks


@st.cache_resource(show_spinner="Building FAISS knowledge base...")
def build_knowledge_base(pdf_bytes: bytes, file_hash: str):
    pages = extract_pdf_text(pdf_bytes)

    if not pages:
        raise ValueError(
            "No readable text was extracted from this PDF. "
            "If it is a scanned PDF, OCR may be required."
        )

    chunks = chunk_pages(pages)

    texts = [item["text"] for item in chunks]
    embeddings = embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return {
        "index": index,
        "chunks": chunks,
        "pages": len(pages),
        "chunks_count": len(chunks),
        "file_hash": file_hash,
    }


def retrieve(query, knowledge_base, top_k=5):
    query_embedding = embedder.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = np.asarray(query_embedding, dtype="float32")

    scores, indices = knowledge_base["index"].search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        item = knowledge_base["chunks"][int(idx)]
        results.append(
            {
                "text": item["text"],
                "page": item["page"],
                "score": float(score),
            }
        )

    return results


# -----------------------------
# Voice transcription
# -----------------------------
def transcribe_audio(audio_file):
    audio_bytes = audio_file.getvalue()

    # Groq accepts file-like objects / uploaded audio.
    transcription = client.audio.transcriptions.create(
        file=("question.wav", audio_bytes),
        model=STT_MODEL,
        response_format="json",
        temperature=0.0,
    )

    return transcription.text.strip()


# -----------------------------
# LLM answer
# -----------------------------
def generate_answer(
    question,
    retrieved_chunks,
    technicality,
    response_size,
    language,
    response_style,
):
    context_parts = []

    for item in retrieved_chunks:
        context_parts.append(
            f"[Page {item['page']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    if language == "English":
        language_instruction = "Answer in clear English."
    elif language == "Urdu":
        language_instruction = "Answer in clear Urdu script."
    else:
        language_instruction = "Answer in easy Roman Urdu."

    system_prompt = f"""
You are a Pakistan Cyber Law RAG Assistant.

Your job is to answer questions about Pakistani cyber/electronic-crime law
USING ONLY the retrieved material supplied from the user's uploaded legal PDF.

IMPORTANT GROUNDING RULES:
1. Do not invent sections, penalties, definitions, authorities, dates, or legal rules.
2. Do not rely on your general model knowledge when the uploaded document does not support a claim.
3. If the retrieved material is insufficient, explicitly say:
   "The uploaded legal document does not provide enough information to answer this reliably."
4. If the document contains a relevant Act/section, identify it when the retrieved text supports it.
5. Distinguish clearly between what the document says and any general explanation.
6. This is legal information, not a substitute for advice from a qualified lawyer.
7. Never help the user commit, conceal, evade, or optimize a cybercrime.
8. For potentially harmful cyber questions, keep the response defensive/legal and explain the relevant law instead of giving operational attack instructions.

User preferences:
- Technicality: {technicality}
- Response size: {response_size}
- Language: {language}
- Style: {response_style}

{language_instruction}

Return a useful, structured answer with:
- Direct answer
- Relevant law/section, ONLY if supported by the retrieved context
- Simple explanation
- Practical legal significance, if supported
- Short disclaimer where appropriate
"""

    user_prompt = f"""
Retrieved legal context:

{context}

User question:
{question}
"""

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=2200 if response_size == "Detailed" else 1200,
    )

    return completion.choices[0].message.content


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("⚙️ Assistant Settings")

    technicality = st.selectbox(
        "Technicality level",
        ["Beginner", "Intermediate", "Legal / Technical"],
        index=1,
    )

    response_size = st.selectbox(
        "Response size",
        ["Short", "Medium", "Detailed"],
        index=1,
    )

    language = st.selectbox(
        "Answer language",
        ["English", "Urdu", "Roman Urdu"],
        index=0,
    )

    response_style = st.selectbox(
        "Response style",
        [
            "Simple Explanation",
            "Legal Analysis",
            "Section-by-Section",
            "Practical Example",
        ],
        index=0,
    )

    top_k = st.slider(
        "Retrieved legal passages",
        min_value=3,
        max_value=8,
        value=5,
    )

    st.divider()

    st.caption("Models")
    st.code(
        f"LLM: {LLM_MODEL}\n"
        f"Voice → Text: {STT_MODEL}\n"
        f"Embeddings: {EMBED_MODEL_NAME}"
    )

    st.info(
        "Upload the cyber-law PDF in this sidebar. "
        "The app creates a FAISS index from that document."
    )

    uploaded_pdf = st.file_uploader(
        "📄 Upload Pakistan Cyber Law PDF",
        type=["pdf"],
        help="Upload the law document you want the assistant to use as its knowledge source.",
    )

# -----------------------------
# Build knowledge base
# -----------------------------
knowledge_base = None

if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.getvalue()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    try:
        knowledge_base = build_knowledge_base(pdf_bytes, file_hash)

        st.sidebar.success(
            f"Knowledge base ready: "
            f"{knowledge_base['pages']} pages / "
            f"{knowledge_base['chunks_count']} passages"
        )
    except Exception as exc:
        st.error(f"Could not process the PDF: {exc}")
        st.stop()
else:
    st.warning(
        "📄 Please upload your Pakistan cyber-law PDF from the sidebar before asking a question."
    )

# -----------------------------
# Input mode
# -----------------------------
input_mode = st.radio(
    "Choose how you want to ask your question",
    ["⌨️ Type", "🎙️ Voice"],
    horizontal=True,
)

question = ""

if input_mode == "⌨️ Type":
    question = st.text_area(
        "Your legal question",
        placeholder=(
            "Example: What does the law say about unauthorized access "
            "to a computer system?"
        ),
        height=120,
    )

else:
    audio = st.audio_input(
        "🎙️ Record your question",
        sample_rate=16000,
    )

    if audio is not None:
        st.audio(audio)

        if st.button("Convert Voice to Text", type="secondary"):
            with st.spinner("Transcribing your question..."):
                try:
                    question = transcribe_audio(audio)
                    st.session_state["voice_question"] = question
                except Exception as exc:
                    st.error(f"Voice transcription failed: {exc}")

    question = st.session_state.get("voice_question", "")

    if question:
        st.text_area(
            "Transcribed question",
            value=question,
            height=100,
            disabled=True,
        )

# -----------------------------
# Ask
# -----------------------------
ask = st.button(
    "⚖️ Ask Pakistan Cyber Law Assistant",
    type="primary",
    use_container_width=True,
)

if ask:
    if knowledge_base is None:
        st.error("Please upload the cyber-law PDF first.")
        st.stop()

    if not question.strip():
        st.error("Please enter or record a question first.")
        st.stop()

    with st.spinner("Searching the legal document..."):
        retrieved = retrieve(
            question.strip(),
            knowledge_base,
            top_k=top_k,
        )

    if not retrieved:
        st.warning("No relevant passage was found in the uploaded document.")
        st.stop()

    with st.spinner("Generating grounded legal explanation..."):
        try:
            answer = generate_answer(
                question=question.strip(),
                retrieved_chunks=retrieved,
                technicality=technicality,
                response_size=response_size,
                language=language,
                response_style=response_style,
            )
        except Exception as exc:
            st.error(f"Could not generate the answer: {exc}")
            st.stop()

    st.subheader("Answer")
    st.markdown(answer)

    st.divider()
    st.subheader("📚 Retrieved legal sources")

    for i, item in enumerate(retrieved, start=1):
        with st.expander(
            f"Source {i} — PDF page {item['page']} — similarity {item['score']:.3f}"
        ):
            st.write(item["text"])

    st.caption(
        "Grounding note: the answer was generated from passages retrieved "
        "from the uploaded PDF. Always verify important legal matters with "
        "the current official law and a qualified legal professional."
    )
