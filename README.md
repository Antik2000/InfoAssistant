# InfoAssistBot

InfoAssistBot is an interactive research assistant built with Streamlit and LangChain. It allows users to upload PDFs or enter URLs, then uses OpenAI's GPT models to create embeddings and answer questions based on the provided documents.

---

## Features

- Upload multiple PDFs or enter URLs as data sources.
- Automatically processes and splits large documents into chunks.
- Creates vector embeddings for efficient semantic search.
- Conversational retrieval using OpenAI GPT (gpt-3.5-turbo-instruct).
- Maintains chat history for follow-up questions.
- Saves FAISS vector store locally for quick reuse.
- Simple and clean Streamlit user interface.

---
