import streamlit as st
import os
import pickle
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever

load_dotenv()
DB_PATH = "./Database"
CHUNKS_FILE = "./Database/chunks.pkl"
PROMPT_FILE = "prompt.txt"

gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    os.environ["GOOGLE_API_KEY"] = gemini_api_key

st.set_page_config(page_title="WHO Mental Health RAG", page_icon="🧠", layout="wide")
st.title("🧠 WHO Mental Health AI Assistant")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("❌ Error: GEMINI_API_KEY not found in .env file.")
    st.stop()


@st.cache_resource
def get_embedding_function():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource
def get_vector_store(_embedding_function):
    return Chroma(persist_directory=DB_PATH, embedding_function=_embedding_function)


class HybridRetriever:
    def __init__(self, vector_ret, keyword_ret, vector_weight=0.6, keyword_weight=0.4):
        self.vector_ret = vector_ret
        self.keyword_ret = keyword_ret
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    # RRF
    def rank_fusion(self, vector_docs, keyword_docs, k=60):
        fused_scores = {}
        doc_map = {}

        # Process Vector Results
        for rank, doc in enumerate(vector_docs):
            doc_content = doc.page_content
            doc_map[doc_content] = doc
            if doc_content not in fused_scores:
                fused_scores[doc_content] = 0.0
            fused_scores[doc_content] += self.vector_weight * (1 / (rank + k))

        # Process Keyword Results
        for rank, doc in enumerate(keyword_docs):
            doc_content = doc.page_content
            doc_map[doc_content] = doc
            if doc_content not in fused_scores:
                fused_scores[doc_content] = 0.0
            fused_scores[doc_content] += self.keyword_weight * (1 / (rank + k))

        # Sort by final score
        reranked_results = sorted(
            fused_scores.items(), key=lambda x: x[1], reverse=True
        )

        return [doc_map[content] for content, score in reranked_results]

    def invoke(self, query):
        vector_results = self.vector_ret.invoke(query)
        keyword_results = self.keyword_ret.invoke(query)

        # Fuse and Rerank
        combined_docs = self.rank_fusion(vector_results, keyword_results)

        return combined_docs[:10]


@st.cache_resource
def get_retriever(_embedding_function):
    vectorstore = Chroma(
        persist_directory=DB_PATH, embedding_function=_embedding_function
    )
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

    if not os.path.exists(CHUNKS_FILE):
        st.error("❌ Chunks file not found. Run build_vectorDB.py first!")
        st.stop()

    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 7

    # 60% semantic, 40% keyword
    return HybridRetriever(
        vector_ret=chroma_retriever,
        keyword_ret=bm25_retriever,
        vector_weight=0.6,
        keyword_weight=0.4,
    )


try:
    embedding_function = get_embedding_function()
    retriever = get_retriever(embedding_function)
except Exception as e:
    st.error(f"Error initializing system: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        welcome_message = """Hello! 👋 I'm your WHO Mental Health AI Assistant. 

Feel free to ask me anything related to WHO's mental health at work guidelines, and I'll provide you with accurate, evidence-based information.

**How can I help you today?**"""
    st.markdown(welcome_message)
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})

for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about WHO Mental Health Guidelines..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        results = retriever.invoke(prompt)

        if len(results) == 0:
            response = "Unable to find relevant information."
            sources_display = []
        else:
            context_text = "\n\n---\n\n".join([doc.page_content for doc in results])

            if os.path.exists(PROMPT_FILE):
                with open(PROMPT_FILE, "r") as f:
                    template = f.read()
            else:
                template = """Answer the question based only on the following context:
                {context}
                ---
                Answer the question based on the above context: {question}"""

            prompt_template = template.format(context=context_text, question=prompt)

            model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            response = model.invoke(prompt_template).content

            # Clean Source Extraction
            unique_pages = sorted(
                list(set([str(doc.metadata.get("page", "?")) for doc in results]))
            )
            sources_display = [f"Page {page}" for page in unique_pages]

        st.markdown(response)

        if sources_display:
            with st.expander("View Sources"):
                st.write(", ".join(sources_display))

        st.session_state.messages.append({"role": "assistant", "content": response})
