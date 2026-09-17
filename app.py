import streamlit as st
import os
import tempfile
from io import BytesIO
from gtts import gTTS
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

st.set_page_config(page_title="Cyber Law AI Agent", layout="wide")

# ==========================================
# MODULE 1: AUDIO WORKFLOW (STT & TTS)
# ==========================================
def speech_to_text(audio_bytes, api_key):
    """Converts user voice to text using Groq's fast Whisper model."""
    try:
        client = Groq(api_key=api_key)
        # Streamlit gives us bytes, we need to save temporarily for Groq API
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        with open(temp_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_path, file.read()),
                model="whisper-large-v3",
            )
        os.remove(temp_path)
        return transcription.text
    except Exception as e:
        st.error(f"Speech-to-Text Error: {e}")
        return ""

def text_to_speech(text):
    """Converts AI text response back to voice using gTTS."""
    try:
        # Generate audio in memory (ur for Urdu/English accent blend)
        tts = gTTS(text=text, lang='ur') 
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        st.error(f"Text-to-Speech Error: {e}")
        return None

# ==========================================
# MODULE 2: RAG WORKFLOW (PDF -> FAISS -> LLM)
# ==========================================
@st.cache_resource
def setup_rag_pipeline(pdf_path):
    """Ingests PDF, creates embeddings, and builds FAISS vector store."""
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Free open-source embeddings via HuggingFace
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"RAG Setup Error: {e}")
        return None

def get_answer(query, vectorstore, api_key, tech_level, length_pref):
    """Retrieves context and generates response via Groq."""
    try:
        llm = ChatGroq(api_key=api_key, model_name="llama-3.3-70b-versatile")
        
        system_prompt = f"""You are an AI expert specializing in Pakistan Cyber Laws (PECA).
        Answer the user's question based ONLY on the provided context.
        If the answer is not in the context, say "I cannot find this in the provided Cyber Laws."
        
        Technicality Level: {tech_level}
        Response Length Constraint: {length_pref}
        
        Context: {{context}}
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        response = rag_chain.invoke({"input": query})
        return response["answer"]
    except Exception as e:
        st.error(f"LLM Generation Error: {e}")
        return "Sorry, I encountered an error generating the response."

# ==========================================
# MODULE 3: STREAMLIT UI
# ==========================================
def main():
    st.title("⚖️ Pakistan Cyber Law AI (Voice RAG)")
    st.markdown("Speak your question and the AI will analyze the Cyber Laws to answer you.")
    
    with st.sidebar:
        st.header("⚙️ App Settings")
        
        # --- NEW API KEY LOGIC ---
        # 1. Check Streamlit Cloud Secrets
        # 2. If not found, check Local Environment Variables
        # 3. If still not found, ask user in sidebar
        if "GROQ_API_KEY" in st.secrets:
            groq_api_key = st.secrets["GROQ_API_KEY"]
        elif os.environ.get("GROQ_API_KEY"):
            groq_api_key = os.environ.get("GROQ_API_KEY")
        else:
            groq_api_key = st.text_input("Groq API Key (Required for local run)", type="password", help="Get it free from console.groq.com")
            
        st.subheader("Response Tuning")
        tech_level = st.select_slider("Technicality Level", options=["Layman (Simple)", "Standard", "Legal Expert"])
        length_pref = st.select_slider("Response Size", options=["Short & Concise", "Medium", "Detailed"])
        
        st.subheader("Document Upload")
        uploaded_file = st.file_uploader("Upload Cyber Laws PDF", type="pdf")
    
    if not groq_api_key:
        st.info("👈 API Key not detected in Secrets. Please enter your Groq API Key in the sidebar.")
        return
        
    vectorstore = None
    if uploaded_file:
        # Save uploaded PDF temporarily for the loader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            pdf_path = f.name
            
        with st.spinner("Processing PDF and building Vector Database..."):
            vectorstore = setup_rag_pipeline(pdf_path)
        st.success("Cyber Laws Loaded Successfully!")
    else:
        st.warning("Please upload the Cyber Laws PDF from the sidebar.")
        
    if vectorstore:
        st.divider()
        st.subheader("🎤 Ask Your Question")
        # Native Streamlit audio input
        audio_value = st.audio_input("Record your voice")
        
        if audio_value:
            with st.spinner("Transcribing your voice..."):
                user_text = speech_to_text(audio_value.getvalue(), groq_api_key)
            
            if user_text:
                st.write(f"**You asked:** {user_text}")
                
                with st.spinner("Searching laws and generating AI response..."):
                    answer = get_answer(user_text, vectorstore, groq_api_key, tech_level, length_pref)
                
                st.write(f"**AI Agent:** {answer}")
                
                with st.spinner("Generating Voice Response..."):
                    audio_fp = text_to_speech(answer)
                    if audio_fp:
                        st.audio(audio_fp, format='audio/mp3', autoplay=True)

if __name__ == "__main__":
    main()
