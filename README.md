# ⚖️ Pakistan Cyber Law AI (Voice-to-Voice RAG)

This is a Retrieval-Augmented Generation (RAG) AI agent built using Python and Streamlit. It allows users to upload a PDF (e.g., Pakistan Cyber Laws / PECA), ask questions using their **voice**, and receive answers in both text and **voice**.

## 🚀 Features
- **Voice Input:** Uses Groq's insanely fast Whisper-large-v3 model for Speech-to-Text.
- **Voice Output:** Uses `gTTS` to read the AI's response aloud automatically.
- **Customizable UI:** Control the technicality level (Layman to Expert) and response size directly from the sidebar.
- **Free Stack:** Uses FAISS for local vector storage, HuggingFace for free embeddings (`all-MiniLM-L6-v2`), and the Groq free tier for LLM generation.

## 🛠️ Local Installation

1. Clone this repository:
   ```bash
   git clone <your-repo-link>
   cd <your-repo-folder>
