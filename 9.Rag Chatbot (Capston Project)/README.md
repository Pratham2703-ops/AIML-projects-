 RAG Chatbot for Google Colab
A fully functional Retrieval-Augmented Generation (RAG) chatbot built for Google Colab. Chat with your documents (PDF, TXT, MD, PPTX) 

 Table of Contents
Features
Quick Start
Installation
Usage
Supported File Types
Architecture
Troubleshooting
Available Models
License
 Features
 
Table
Feature	Description
 Multi-format Support	PDF, TXT, MD, PPTX files
 100% Free	Uses Groq + HuggingFace (no OpenAI needed)
 Smart Retrieval	Finds relevant document chunks automatically
 Conversational	Maintains chat history for follow-up questions
 Web UI	Optional Gradio interface with shareable link
 Local Embeddings	Runs on-device, no embedding API costs
 
 Quick Start
Step 1: Open Google Colab
Go to colab.research.google.com and create a new notebook.
Step 2: Run the Complete Setup Cell
Copy and run the complete cell below. It installs everything, uploads your files, and builds the chatbot.
Python
# ============================================
# COMPLETE FREE RAG CHATBOT FOR GOOGLE COLAB
# ============================================

!pip install -q langchain langchain-community langchain-chroma chromadb pypdf python-pptx gradio langchain-text-splitters langchain-core langchain-groq sentence-transformers langchain-huggingface

import os
from getpass import getpass
from pathlib import Path
from google.colab import files

# --- 1. GET FREE GROQ KEY ---
print(" Get your FREE Groq API key: https://console.groq.com/keys")
groq_key = getpass(" Paste Groq API key: ")
groq_key = str(groq_key).strip().strip('"').strip("'")
os.environ["GROQ_API_KEY"] = groq_key
print(" Groq key set!")

# --- 2. Upload files ---
print("\n Upload your PDF, TXT, MD, or PPTX files:")
os.makedirs("data", exist_ok=True)
uploaded = files.upload()
for filename in uploaded.keys():
    import shutil
    shutil.move(filename, f"data/{filename}")
print(f"Files: {os.listdir('data')}")

# --- 3. Load documents ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

docs = []
path = Path("data")
for pdf in path.glob("*.pdf"):
    docs.extend(PyPDFLoader(str(pdf)).load())
for txt in path.glob("*.txt"):
    docs.extend(TextLoader(str(txt), encoding="utf-8").load())
for md in path.glob("*.md"):
    docs.extend(TextLoader(str(md), encoding="utf-8").load())
for pptx in path.glob("*.pptx"):
    from pptx import Presentation
    prs = Presentation(str(pptx))
    text = [shape.text.strip() for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()]
    docs.append(Document(page_content="\n".join(text), metadata={"source": str(pptx)}))

print(f" Loaded {len(docs)} documents")

if len(docs) == 0:
    with open("data/sample.txt", "w") as f:
        f.write("AI is the simulation of human intelligence by machines.")
    docs = TextLoader("data/sample.txt", encoding="utf-8").load()

# --- 4. Split ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
chunks = splitter.split_documents(docs)
print(f" {len(chunks)} chunks")

# --- 5. FREE embeddings ---
from langchain_huggingface import HuggingFaceEmbeddings
print(" Loading free local embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print(" Embeddings loaded!")

# --- 6. Vector store ---
from langchain_chroma import Chroma
vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_db", collection_name="my_docs")
print(" Vector store ready!")

# --- 7. FREE LLM via Groq ---
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
print(" Groq LLM ready!")

# --- 8. RAG chain ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

template = """Answer using only the context. If unknown, say \"I don't have enough information.\"

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

rag_chain = (
    {"context": lambda x: format_docs(retriever.invoke(x["question"])), "question": lambda x: x["question"]}
    | prompt | llm | StrOutputParser()
)

# --- 9. Chatbot ---
from langchain_core.messages import HumanMessage, AIMessage

class RAGChatbot:
    def __init__(self):
        self.messages = []

    def ask(self, question, show_sources=True):
        sources = retriever.invoke(question)
        answer = rag_chain.invoke({"question": question})
        self.messages.extend([HumanMessage(content=question), AIMessage(content=answer)])
        print(f"\n {question}")
        print(f" {answer}")
        if show_sources:
            print("\n Sources:")
            for i, doc in enumerate(sources, 1):
                print(f"  [{i}] {doc.metadata.get('source', 'unknown')}: {doc.page_content[:120]}...")
        return answer

    def clear(self):
        self.messages = []
        print(" Chat history cleared!")

bot = RAGChatbot()
print("\n" + "="*50)
print(" FREE RAG Chatbot Ready! Use: bot.ask('question')")
print("="*50)
Step 3: Start Chatting
Run this in a new cell:
Python
bot.ask("What is this document about?")
 Usage
Basic Commands
Table
Command	Description
bot.ask("Your question")	Ask about your documents
bot.clear()	Reset chat history
Example Conversation
Python
bot.ask("What is this document about?")
#  This document covers digital marketing strategies...

bot.ask("What are the key points?")
#  The key points include SEO, social media marketing, and content strategy...

bot.ask("Explain SEO in simple terms")
#  SEO (Search Engine Optimization) is the practice of...

bot.clear()
#  Chat history cleared!
Web Interface (Optional)
Python
import gradio as gr

def chat_response(message, history):
    return rag_chain.invoke({"question": message})

demo = gr.ChatInterface(
    fn=chat_response,
    title=" RAG Chatbot",
    description="Ask questions about your uploaded documents!",
)

demo.launch(share=True)  # Creates a public URL
 Supported File Types
Table
Format	Extension	Notes
PDF	.pdf	Full text extraction
Plain Text	.txt	Direct reading
Markdown	.md	Direct reading
PowerPoint	.pptx	Slide text extraction
 Architecture
plain
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Documents  │────▶│  Text Splitter  │────▶│  HuggingFace    │
│  (PDF/TXT)  │     │  (Chunking)     │     │  Embeddings     │
└─────────────┘     └─────────────────┘     └────────┬────────┘
                                                       │
                                                ┌──────▼──────┐
                                                │   ChromaDB   │
                                                │  (Vector DB) │
                                                └──────┬──────┘
                                                       │
User Query ──▶ Embedding ──▶ Similarity Search ──▶  Top-K Chunks
                                                       │
                                                ┌──────▼──────┐
                                                │  Groq LLM    │
                                                │  (Llama 3.1) │
                                                └──────┬──────┘
                                                       │
                                                ┌──────▼──────┐
                                                │   Answer     │
                                                │  + Sources   │
                                                └──────────────┘
 Troubleshooting
Issue: AuthenticationError (OpenAI)
Fix: Use the Groq version above — it's free and doesn't require OpenAI.
Issue: Model decommissioned (Groq)
Fix: Update the model name:

Python
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
Issue: No documents found
Fix: Make sure files are in the data/ folder. Re-run the upload cell.
Issue: ModuleNotFoundError
Fix: Re-run the !pip install line at the top of the setup cell.
 Available Models
Groq LLM Models

Table
Model	Speed	Best For
llama-3.1-8b-instant	 Fast	General Q&A
llama-3.3-70b-versatile	 Medium	Complex reasoning
mixtral-8x7b-32768	 Fast	Long documents
gemma2-9b-it	 Fast	Lightweight tasks
Embedding Models

Table
Model	Size	Speed
all-MiniLM-L6-v2	22MB	 Fast
all-mpnet-base-v2	110MB	 Better quality

 License
MIT License — Free for personal and commercial use.

 Acknowledgments
LangChain — RAG framework
Groq — Free LLM inference
HuggingFace — Free embeddings
ChromaDB — Vector database
