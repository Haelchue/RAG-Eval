
# RAG+EVAL
### An Evaluated, Metrics-Driven RAG System for Reliable PDF Question Answering

> **An end-to-end, evaluated Retrieval-Augmented Generation (RAG) system for question answering over PDF documents, with retrieval and generation metrics.**

---

## 🚀 Project Overview

This project implements a **production-style Retrieval-Augmented Generation (RAG) pipeline** that allows users to ask **natural language questions over PDF documents** and receive **accurate, context-grounded answers**.

Unlike simple LLM-based chatbots, this system:

* Retrieves relevant document chunks using **vector search**
* Improves relevance using **reranking**
* Generates answers grounded strictly in retrieved context
* **Quantitatively evaluates** both retrieval quality and answer quality

## 🧠 Why RAG?

Large Language Models (LLMs) alone:

* ❌ Hallucinate facts
* ❌ Cannot access private documents
* ❌ Are not verifiable

Retrieval-Augmented Generation (RAG):

* ✅ Grounds answers in external documents
* ✅ Scales to large corpora
* ✅ Reduces hallucinations
* ✅ Enables evaluation

This project demonstrates **RAG the correct way — with metrics**.

---

## 🏗️ System Architecture

```
PDF Documents
      ↓
PDF Loader (PyPDF)
      ↓
Recursive Text Chunking
      ↓
Sentence Transformer Embeddings
      ↓
FAISS Vector Store
      ↓
Retriever
      ↓
Cross-Encoder Reranker
      ↓
LLM (Groq)
      ↓
Final Answer
```

---

## 🔧 Tech Stack

### Core Technologies

* **Python**
* **LangChain**
* **FAISS** – Vector similarity search
* **Sentence Transformers** – Dense embeddings
* **Cross-Encoder** – Reranking retrieved chunks
* **Groq LLM API** – Fast LLM inference

### Supporting Libraries

* `pypdf`
* `transformers`
* `torch`
* `numpy`
* `scikit-learn`

---

## ✨ Key Features

* 📄 PDF ingestion and parsing
* ✂️ Recursive text chunking with overlap
* 🧬 Dense vector embeddings
* 🔍 FAISS-based similarity search
* 🎯 Cross-encoder reranking (**advanced RAG**)
* 🤖 LLM-based answer generation
* 📊 **Retrieval evaluation (Recall@K, MRR)**
* 🧠 **Answer quality evaluation (ROUGE, BLEU, BERTScore)**
* 💾 Metrics saved as JSON & CSV

> ⚠️ Most student RAG projects **do not include evaluation**.
> This project explicitly measures performance.

---

## 🧪 How It Works (Technical Flow)

1. **PDF Loading**

   * PDFs are loaded using `PyPDFLoader`.

2. **Text Chunking**

   * Documents are split into overlapping chunks to preserve semantic continuity.

3. **Embedding Generation**

   * Sentence Transformers convert text chunks into dense vector representations.

4. **Vector Storage**

   * FAISS indexes embeddings for fast similarity search.

5. **Retrieval**

   * Top-k relevant chunks are retrieved for a given query.

6. **Reranking**

   * A cross-encoder reranks retrieved chunks by semantic relevance.

7. **Answer Generation**

   * Top-ranked chunks are injected into an LLM prompt to generate grounded answers.

---

## 📊 Evaluation & Metrics

This project includes **quantitative evaluation** for both **retrieval quality** and **answer quality**, following best practices in RAG system design.

---

### 🔍 Retrieval Evaluation (FAISS-only, LLM-independent)

| Metric    | Value  |
| --------- | ------ |
| Recall@1  | 0.55   |
| Recall@3  | 0.775  |
| Recall@5  | 0.80   |
| Recall@10 | 0.85   |
| MRR       | 0.6587 |

**Interpretation:**

* Relevant document chunks are retrieved within the top-5 results in most cases
* High MRR indicates that correct context appears early in rankings

---

### 🧠 Answer Quality Evaluation (LLM Output)

| Metric         | Score |
| -------------- | ----- |
| ROUGE-1 F1     | 0.616 |
| ROUGE-2 F1     | 0.376 |
| ROUGE-L F1     | 0.557 |
| BLEU           | 16.55 |
| BERTScore (F1) | 0.659 |

**Interpretation:**

* Strong semantic similarity between generated and reference answers
* Confirms factual grounding via retrieved context

---

📁 Metrics are automatically saved as:

* `answer_quality_metrics.json`
* `answer_quality_metrics.csv`
* `retrieval_metrics_faiss.json`

---

## 📂 Project Structure

```
├── evaluated_rag_pdf_qa.ipynb
├── data/
│   └── pdfs/
├── metrics/
│   ├── answer_quality_metrics.json
│   ├── answer_quality_metrics.csv
│   └── retrieval_metrics_faiss.json
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Setup & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/evaluated-rag-pdf-qa.git
cd evaluated-rag-pdf-qa
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Set Environment Variables

```bash
export GROQ_API_KEY="your_api_key_here"
```

> ⚠️ **Never hardcode API keys inside notebooks or source files**

---

### 4️⃣ Run the Notebook

Open `evaluated_rag_pdf_qa.ipynb` and execute cells sequentially.

---

## 📌 Use Cases

* Academic paper analysis
* Legal and policy document QA
* Research assistants
* Enterprise knowledge-base chatbots
* Internal document search systems

---

## 🔮 Future Improvements

* Conversational memory
* Source citation per answer
* Hybrid search (BM25 + vector search)
* Streamlit / FastAPI interface
* RAG evaluation with RAGAS
* Persistent vector database

---

## 🧠 Research & Engineering Highlights

* Separates **retrieval evaluation** from **generation evaluation**
* Uses **reranking** to improve retrieval precision
* Avoids hallucinations via strict context grounding
* Follows practices used in **enterprise and research RAG systems**

---

## 👤 Author

**Hillsea Rana**
AI / ML Enthusiast
Interests: LLMs, RAG systems, Computer Vision 

---

## 📜 License

This project is licensed under the **MIT License**.

---


Just tell me what’s next.
