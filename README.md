# 📄 Document Q&A App (LangChain + Streamlit)

This project is a **Document Question-Answering App** built using **LangChain**, **Streamlit**, and **Groq's Gemma 2 9B** language model. It allows you to upload a **PDF** or **Word document**, and then interact with it via natural language chat to get intelligent, context-aware answers.

---

## 🚀 Features

- 🧠 Question Answering from uploaded documents (PDF or DOCX)
- 🔍 Retrieval Augmented Generation (RAG) using **ChromaDB** + **MiniLM Embeddings**
- 💬 Interactive chat with memory (Streamlit chat)
- ⚙️ Streamlit interface for ease of use
- 🪄 Fast and accurate responses using **Gemma 2 9B** on **Groq API**

---

## 📁 Project Structure

```
├── app.py                    # Main Streamlit app
├── .env                      # Store API keys securely
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/document-qa-app.git
cd document-qa-app
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add environment variables

Create a `.env` file with the following:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## 🧪 Run the App

```bash
streamlit run app.py
```

Then open the URL Streamlit provides (usually http://localhost:8501) in your browser.

---

## 🛠 Tech Stack

- **[Streamlit](https://streamlit.io/)** - Web interface
- **[LangChain](https://python.langchain.com/)** - LLM orchestration and RAG
- **[Chroma](https://docs.trychroma.com/)** - Vector store for document retrieval
- **[HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)** - Text embeddings
- **[Groq API](https://groq.com/)** - LLM inference using Gemma 2 (9B)
- **[Python-dotenv](https://pypi.org/project/python-dotenv/)** - For managing environment variables

---

## 🧠 How It Works

1. Upload a **PDF** or **DOCX** file.
2. The document is split into chunks.
3. Each chunk is embedded using MiniLM and stored in **Chroma vector DB**.
4. User input is matched against document chunks using **similarity search**.
5. The **Gemma-2 9B** LLM generates an answer using both user query and relevant context.

---

## 📌 Example Use Cases

- Chat with legal contracts 📜
- Summarize reports 📊
- Search academic papers 📚
- Extract info from HR policies 🏢

---

## 📃 License

MIT License — feel free to use and modify.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you would like to change.

---

## 🙋‍♂️ Author

Made with ❤️ by [Your Name](https://github.com/yourusername)

```

---

Let me know if you'd like a version with a logo, screenshots, or GitHub badges.
