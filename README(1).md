# 🇵🇰 Pakistan Cyber Law AI Assistant — Voice + RAG

A Streamlit RAG application for asking questions about a Pakistan cyber-law PDF.

The app supports:

- 📄 PDF-based legal knowledge base
- 🔎 FAISS semantic retrieval
- 🧠 Groq `openai/gpt-oss-120b` for answer generation
- 🎙️ Microphone questions using Streamlit `st.audio_input`
- 🗣️ Groq `whisper-large-v3-turbo` for voice-to-text
- ⌨️ Normal text questions
- 🎚️ Technicality level
- 📏 Response size
- 🌐 English / Urdu / Roman Urdu
- ⚖️ Simple explanation / legal analysis / section-by-section / practical example
- 📚 Retrieved source passages with PDF page numbers
- 🔐 Groq API key stored as a Streamlit Secret

## Files

Only these three files are required:

```text
app.py
requirements.txt
README.md
```

## 1. Get a Groq API key

Create a Groq API key and keep it private.

Do **not** put the key directly inside `app.py`.

## 2. Run locally / Google Colab

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```

If using Google Colab, you can install the requirements and run Streamlit with a suitable tunnel if you want to test it through a browser.

## 3. Streamlit Cloud deployment

Push these three files to a GitHub repository:

```text
app.py
requirements.txt
README.md
```

Then create a new Streamlit app from that GitHub repository.

In Streamlit:

```text
App → Settings / Manage app → Secrets
```

Add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Save the secret and restart/redeploy the app.

## 4. Upload the cyber-law PDF

This version intentionally does **not** hard-code a fourth PDF file into the repository.

Instead, the app has:

```text
Sidebar → Upload Pakistan Cyber Law PDF
```

Upload the cyber-law PDF that you want the RAG assistant to use.

The app extracts the PDF text, creates chunks, generates embeddings, and builds a FAISS index in memory.

This keeps the GitHub project at exactly three files.

## 5. How voice works

The voice pipeline is:

```text
🎙️ Microphone
      ↓
Streamlit st.audio_input()
      ↓
Groq Whisper
whisper-large-v3-turbo
      ↓
Question text
      ↓
FAISS retrieval
      ↓
Relevant PDF passages
      ↓
Groq GPT-OSS 120B
      ↓
⚖️ Legal answer
```

The application does **not** require VAPI, ElevenLabs, Retell, or another voice-agent platform for voice input.

## 6. Important legal grounding behavior

The assistant is deliberately instructed to use the uploaded PDF as its legal knowledge source.

It should:

- identify relevant sections only when supported by retrieved PDF text;
- avoid inventing legal provisions or penalties;
- say when the uploaded document does not contain enough information;
- provide legal information rather than pretending to be a lawyer;
- avoid giving operational instructions for committing or concealing cybercrime.

## 7. Important note about the PDF

For the best results, upload a searchable/text-based PDF.

If the PDF is only scanned images, `pypdf` may not be able to extract its text. OCR would then be required.

## 8. Models

### Answer model

```text
openai/gpt-oss-120b
```

### Voice-to-text model

```text
whisper-large-v3-turbo
```

### Embedding model

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding model is downloaded automatically when the app first starts.

## 9. Voice output

This version intentionally focuses on:

**voice input → RAG → text answer**

That is the simplest and most deployment-friendly architecture.

A text-to-speech layer can be added later if you want the assistant to read the legal answer aloud.

## 10. Security

Never commit:

```text
GROQ_API_KEY
```

to GitHub.

Use Streamlit Secrets instead.

Also remember that a public Streamlit app using your API key can consume your API quota. For a public production application, authentication, rate limiting, and usage controls should be added.

## 11. Disclaimer

This project is an educational/legal-information RAG assistant. It is not a lawyer and does not replace professional legal advice. Laws and amendments can change, so important legal matters should be checked against the current official legal text and qualified legal counsel.
