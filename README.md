# ScholarMate

![ScholarMate Banner](images/ScholarMate.png)

**ScholarMate** is an AI-powered academic assistant that helps students engage with their learning materials more efficiently. It uses Streamlit and LangChain with Groq’s LLaMA 3 model to provide features like document-based question answering, summarization, MCQ generation, and topic-wise explanations — all through an intuitive web interface.

---

## 📌 Features

- **Upload & Process Files**: Accepts `.pdf`, `.docx`, and `.txt` formats.
- **Smart Q&A**: Ask questions directly based on the uploaded documents.
- **Summarization**: Generate concise summaries from your academic content.
- **MCQ Generation**: Automatically creates multiple-choice questions.
- **Topic-wise Explanation**: Get detailed explanations for specific topics.
- **Confidence Score**: Each answer includes a semantic confidence score.

---

## 🖼️ Screenshots

| Logo                     | Banner                          |
|--------------------------|----------------------------------|
| ![Logo](images/logo.png)  | ![Banner](images/ScholarMate.png) |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ScholarMate.git
cd ScholarMate
```

### 2. Set Up a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Create `.env` File

Create a `.env` file in the root directory and add the following:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> **Important:** Make sure your `.env` file is listed in `.gitignore` so that it never gets pushed to your GitHub repository.

---

## 💻 Run the App

```bash
streamlit run main.py
```

---

## 🌍 Deployment (Streamlit Cloud)

1. Push your code to a GitHub repository.
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud).
3. Click “New App” and select your GitHub repo.
4. Set `main.py` as the entry point.
5. Add the `GROQ_API_KEY` as a secret variable under “Advanced settings”.

---

## 📂 Project Structure

```
ScholarMate/
├── image/
│   ├── logo.png
│   └── ScholarMate.png
├── .gitignore
├── main.py
├── README.md
├── requirements.txt
```

---

## 🛠️ Built With

- **[Streamlit](https://streamlit.io/)** — Fast interactive apps in Python
- **[LangChain](https://www.langchain.com/)** — Framework for LLM applications
- **[FAISS](https://github.com/facebookresearch/faiss)** — Vector search engine
- **[Groq API (LLaMA 3)](https://console.groq.com/)** — Powerful LLM backend
- **[HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)** — For document similarity

---

## 🙈 .gitignore Reminder

Make sure your `.gitignore` file includes the following:

```
.env
__pycache__/
*.pyc
```

This protects your API keys and avoids unnecessary files in Git.

---

## 📬 Feedback & Contributions

Feel free to open issues or pull requests if you'd like to contribute or suggest improvements!

---
