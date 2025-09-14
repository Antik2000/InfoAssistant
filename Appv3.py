import os
import streamlit as st
import time
from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI
from dotenv import load_dotenv

# Load local .env file (for local testing)
load_dotenv()

# Get OpenAI API key from Streamlit secrets or .env
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

if not OPENAI_API_KEY:
    st.error("❌ No OpenAI API key found. Please set it in .streamlit/secrets.toml or your .env file.")
else:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Streamlit UI setup
st.set_page_config(page_title="Infobot 📄🔍", page_icon="🧠")
st.title("Infobot: News & PDF Research Assistant 🧠")
st.sidebar.title("📥 Data Inputs")

# Sidebar: URLs & PDFs
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
uploaded_pdfs = st.sidebar.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
process_inputs_clicked = st.sidebar.button("⚙️ Process Inputs")

# FAISS storage path
faiss_dir = "faiss_store_openai"
main_placeholder = st.empty()

# Initialize OpenAI LLM
llm = OpenAI(
    temperature=0.8,  # Creative responses
    max_tokens=1500,
    model_name="gpt-3.5-turbo-instruct"
)

# Session memory (for follow-up questions)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

# Process documents on button click
if process_inputs_clicked:
    documents = []

    # Load data from URLs
    urls = [u.strip() for u in urls if u.strip()]
    if urls:
        url_loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("🔄 Loading data from URLs...")
        documents.extend(url_loader.load())

    # Load data from PDFs
    if uploaded_pdfs:
        for pdf in uploaded_pdfs:
            pdf_path = f"temp_{pdf.name}"
            with open(pdf_path, "wb") as f:
                f.write(pdf.getbuffer())
            pdf_loader = PyPDFLoader(pdf_path)
            main_placeholder.text(f"🔄 Loading {pdf.name}...")
            documents.extend(pdf_loader.load())

    # Warn if nothing is uploaded
    if not documents:
        st.warning("⚠️ No valid documents provided.")
    else:
        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("✂️ Splitting text into chunks...")
        docs = text_splitter.split_documents(documents)

        # Create embeddings & vector store
        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("🔍 Creating vectorstore with embeddings...")
        time.sleep(1)

        # Save FAISS index locally
        vectorstore_openai.save_local(faiss_dir)
        main_placeholder.success("✅ Data indexed successfully!")

# Chat input
query = st.text_input("💬 Ask a question (you can follow up):")

if query:
    if os.path.exists(faiss_dir):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            faiss_dir, embeddings, allow_dangerous_deserialization=True
        )

        # Conversational Retrieval Chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=st.session_state.memory,
            verbose=False
        )

        result = chain({"question": query})

        # Show the answer
        st.subheader("🧠 Answer")
        st.write(result["answer"])

        # Show chat history
        with st.expander("📝 Chat History"):
            messages = st.session_state.memory.chat_memory.messages
            for i, msg in enumerate(messages):
                role = "User" if i % 2 == 0 else "Bot"
                st.markdown(f"**{role}:** {msg.content}")
