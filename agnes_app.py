# agnes_app.py
from openai import OpenAI
import streamlit as st
import openai
import fitz  # PyMuPDF
import faiss
import numpy as np

# === SETTINGS ===
openai.api_key = st.secrets["OPENAI_API_KEY"]  # You'll add this later in the cloud

# === Load PDF ===
@st.cache_data
def load_book(path="agnes_book.pdf"):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return response.data[0].embedding


@st.cache_resource
def build_vector_index(chunks):
    dim = 1536
    index = faiss.IndexFlatL2(dim)
    meta = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        index.add(np.array([emb], dtype="float32"))
        meta.append(chunk)
    return index, meta

def search_index(query, index, metadata, top_k=3):
    q_emb = get_embedding(query)
    D, I = index.search(np.array([q_emb], dtype="float32"), top_k)
    return [metadata[i] for i in I[0]]

def ask_agnes(question, context_chunks):
    context = "\n\n".join(context_chunks)
    messages = [
        {"role": "system", "content": "You are Agnes, a wise and warm life coach. Use this book content to help."},
        {"role": "user", "content": f"{context}\n\nQuestion: {question}"}
    ]

    client = openai.OpenAI()  # This creates the proper client object for v1+
    response = client.chat.completions.create(
        model="gpt-4",  # or use "gpt-3.5-turbo" if you want faster replies
        messages=messages
    )
    return response.choices[0].message.content


# === STREAMLIT UI ===
st.title("🌸 Talk to Agnes - Your Life Coach")

# Create a chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask Agnes a life question:")

if user_input:
    with st.spinner("Agnes is thinking..."):
        book = load_book()
        chunks = split_text(book)
        index, meta = build_vector_index(chunks)
        context = search_index(user_input, index, meta)
        answer = ask_agnes(user_input, context)

        # Add to session history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Agnes", answer))

# Display chat history
if st.session_state.chat_history:
    for sender, msg in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"**👤 You:** {msg}")
        else:
            st.markdown(f"**🧠 Agnes:** {msg}")
