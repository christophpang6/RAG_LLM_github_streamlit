import os
import streamlit as st
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ======== Sample Training Documents ========
enhanced_sample_texts = {
    "space_missions.txt": """
    The Apollo 11 mission launched on July 16, 1969, and landed the first humans on the Moon on July 20, 1969.
    The crew consisted of exactly three astronauts: Neil Armstrong (Commander), Buzz Aldrin (Lunar Module Pilot),
    and Michael Collins (Command Module Pilot). Neil Armstrong was the first person to walk on the Moon,
    followed by Buzz Aldrin. Michael Collins remained in lunar orbit aboard the command module Columbia.
    The mission lasted 8 days, 3 hours, 18 minutes, and 35 seconds. There was no fourth crew member on Apollo 11.
    """,
    "landmarks_architecture.txt": """
    The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France.
    Construction began in 1887 and was completed in 1889 for the 1889 World's Fair.
    """,
    "programming_technologies.txt": """
    Python was created by Guido van Rossum and first released in 1991.
    It emphasizes code readability with its notable use of significant whitespace.
    """,
    "science_discoveries.txt": """
    Penicillin was discovered by Alexander Fleming in 1928 when he noticed that a mold had killed bacteria in his lab.
    """,
    "historical_events.txt": """
    World War II lasted from 1939 to 1945 and involved most of the world's nations.
    The war ended with the surrender of Germany on May 8, 1945 (Victory in Europe Day)
    and Japan on August 15, 1945, following the atomic bombings of Hiroshima and Nagasaki.
    """
}

# ======== Prepare FAISS Index ========
embedder = SentenceTransformer("all-MiniLM-L6-v2")
corpus, sources = [], []
for src, text in enhanced_sample_texts.items():
    for line in text.strip().split("\n"):
        line = line.strip()
        if line:
            corpus.append(line)
            sources.append(src)

embeddings = embedder.encode(corpus, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# ======== System message enforcing detailed "I don't know" fallback ========
SYSTEM_MESSAGE = (
    "You are a helpful assistant. Only answer based on the provided context. "
    "If the context does not contain the answer, respond with: "
    "'I don't know is used to demonstrate that the chatbot will not hallucinate "
    "if it doesn't know based off of the retrieved context.'"
)

# ======== Streamlit UI Config ========
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Multi-Turn RAG Chatbot Demo")
st.markdown("Ask questions about **space missions, landmarks, programming, science, or historical events.**")

# Hugging Face token input
hf_token = st.text_input("Enter your HuggingFace Token:", type="password")

# Slider controls (like in Gradio)
max_tokens = st.slider("Max new tokens", min_value=1, max_value=2048, value=512, step=1)
temperature = st.slider("Temperature", min_value=0.0, max_value=4.0, value=0.0, step=0.1)
top_p = st.slider("Top-p (nucleus sampling)", min_value=0.1, max_value=1.0, value=0.95, step=0.05)

# Initialize chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ======== RAG + HF Chat Function ========
def rag_respond(message, hf_token, history, system_message, max_tokens, temperature, top_p):
    client = InferenceClient(token=hf_token, model="openai/gpt-oss-20b")

    # Combine previous Q&A
    history_text = ""
    for q, a in history[-3:]:
        history_text += f"Previous Q: {q}\nPrevious A: {a}\n"

    # FAISS retrieval
    retrieval_query = " ".join([f"{q} {a}" for q, a in history[-3:]] + [message])
    q_emb = embedder.encode([retrieval_query], convert_to_numpy=True)
    D, I = index.search(q_emb, k=5)
    retrieved_chunks = [(corpus[i], sources[i], D[0][j]) for j, i in enumerate(I[0])]
    context_text = "\n".join([f"[{src}] {chunk}" for chunk, src, _ in retrieved_chunks])

    # Build messages for HF API
    messages_list = [{"role": "system", "content": system_message}]
    for q, a in history:
        messages_list.append({"role": "user", "content": q})
        messages_list.append({"role": "assistant", "content": a})
    messages_list.append({"role": "user", "content": f"{history_text}\nCurrent Query: {message}\nContext:\n{context_text}"})

    # Collect response
    response_text = ""
    for chunk in client.chat_completion(messages_list, max_tokens=max_tokens, stream=True, temperature=temperature, top_p=top_p):
        if len(chunk.choices) and chunk.choices[0].delta.content:
            response_text += chunk.choices[0].delta.content

    return response_text

# ======== Streamlit Chat Loop ========
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

if prompt := st.chat_input("Ask me something..."):
    if not hf_token:
        st.warning("Please enter your HuggingFace token above first.")
    else:
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_respond(prompt, hf_token, st.session_state.chat_history,
                                       SYSTEM_MESSAGE, max_tokens, temperature, top_p)
                st.write(response)
        st.session_state.chat_history.append((prompt, response))
