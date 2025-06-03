import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.tools import tool
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
import pytz
import os
from PIL import Image
import base64
import requests
from datetime import datetime
import pytz

load_dotenv()

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="ScholarMate",
    layout="wide",
    page_icon=":mortar_board:",
    initial_sidebar_state="expanded"
)

# -------------------- Time-Based Greeting --------------------
def get_user_timezone():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        return data.get("timezone", "UTC")
    except:
        return "UTC"

def get_greeting():
    user_timezone = get_user_timezone()
    now = datetime.now(pytz.timezone(user_timezone))
    hour = now.hour

    if 4 <= hour < 12:
        return "Good Morning! Ready to learn?"
    elif 12 <= hour < 16:
        return "Good Afternoon! Let's study!"
    else:
        return "Good Evening! Time for knowledge!"

# -------------------- Theme System --------------------
def apply_theme(theme_name, dark_mode):
    """Applies the selected theme with dark mode support"""
    themes = {
        'academic': {
            'primary': '#4CAF50',
            'secondary': '#2196F3',
            'background': '#f8f9fa',
            'text': '#2c3e50',
            'card': '#ffffff',
            'input_bg': '#ffffff',
            'input_text': '#000000',
            'button_text': '#ffffff',
            'gradient': 'linear-gradient(135deg, #4CAF50, #2196F3)',
            'sidebar': '#343a40',
            'border': '#dddddd'
        },
        'ocean': {
            'primary': '#2196F3',
            'secondary': '#00BCD4',
            'background': '#E1F5FE',
            'text': '#01579B',
            'card': '#B3E5FC',
            'input_bg': '#E1F5FE',
            'input_text': '#01579B',
            'button_text': '#ffffff',
            'gradient': 'linear-gradient(135deg, #2196F3, #00BCD4)',
            'sidebar': '#01579B',
            'border': '#80DEEA'
        },
        'sunset': {
            'primary': '#FF5722',
            'secondary': '#FF9800',
            'background': '#FFF3E0',
            'text': '#E65100',
            'card': '#FFE0B2',
            'input_bg': '#FFF3E0',
            'input_text': '#E65100',
            'button_text': '#ffffff',
            'gradient': 'linear-gradient(135deg, #FF5722, #FF9800)',
            'sidebar': '#E65100',
            'border': '#FFCC80'
        },
        'forest': {
            'primary': '#4CAF50',
            'secondary': '#8BC34A',
            'background': '#E8F5E9',
            'text': '#1B5E20',
            'card': '#C8E6C9',
            'input_bg': '#E8F5E9',
            'input_text': '#1B5E20',
            'button_text': '#ffffff',
            'gradient': 'linear-gradient(135deg, #4CAF50, #8BC34A)',
            'sidebar': '#1B5E20',
            'border': '#A5D6A7'
        },
        'royal': {
            'primary': '#9C27B0',
            'secondary': '#673AB7',
            'background': '#F3E5F5',
            'text': '#4A148C',
            'card': '#E1BEE7',
            'input_bg': '#F3E5F5',
            'input_text': '#4A148C',
            'button_text': '#ffffff',
            'gradient': 'linear-gradient(135deg, #9C27B0, #673AB7)',
            'sidebar': '#4A148C',
            'border': '#CE93D8'
        }
    }
    
    if dark_mode:
        theme = themes[theme_name].copy()
        theme.update({
            'background': '#121212',
            'text': '#E1E1E1',
            'card': '#1E1E1E',
            'input_bg': '#333333',
            'input_text': '#FFFFFF',
            'sidebar': '#1E1E1E',
            'border': '#444444'
        })
    else:
        theme = themes[theme_name]
    
    return theme

current_theme = apply_theme(st.session_state.theme, st.session_state.dark_mode)

# -------------------- Initialize Session State --------------------
if 'theme' not in st.session_state:
    st.session_state.theme = 'academic'
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
if 'docs' not in st.session_state:
    st.session_state.docs = []
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'summary_output' not in st.session_state:
    st.session_state.summary_output = ""
if 'mcqs_output' not in st.session_state:
    st.session_state.mcqs_output = ""
if 'topic_explanation_output' not in st.session_state:
    st.session_state.topic_explanation_output = ""

# -------------------- Enhanced CSS --------------------
st.markdown(f"""
<style>
/* 🌐 Global App Background */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    background-color: {current_theme['background']} !important;
    color: {current_theme['text']} !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}}

/* 📚 Sidebar */
[data-testid="stSidebar"] {{
    background-color: {current_theme['sidebar']} !important;
    color: white !important;
}}

/* 🎛 Sidebar widgets */
[data-testid="stSidebar"] .stSelectbox, 
[data-testid="stSidebar"] .stExpander, 
[data-testid="stSidebar"] .stSlider, 
[data-testid="stSidebar"] .stToggle {{
    background-color: {current_theme['card']} !important;
    color: {current_theme['text']} !important;
    border-radius: 10px !important;
    padding: 10px !important;
}}

/* 🏷 Brand logo */
.brand-logo {{
    position: absolute;
    top: -65px;
    left: 85px;
    display: flex;
    align-items: center;
    gap: 10px;
    z-index: 1000;
    padding: 5px 10px;
}}

.brand-logo img {{
    height: 60px;
    width: auto;
    border-radius: 10px
}}

.main .block-container {{
    padding-top: 60px;
}}

/* 🧱 Main container */
.main {{
    background-color: {current_theme['background']} !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}}

/* 🔘 Button styles */
.stButton>button {{
    background-color: white !important;
    color: black !important;
    border-radius: 12px !important;
    padding: 10px 20px !important;
    border: 1px solid {current_theme['border']} !important;
    font-weight: 500;
    transition: all 0.3s ease;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}}

.stButton>button:hover {{
    background-color: {current_theme['primary']} !important;
    color: white !important;
    border-color: {current_theme['primary']} !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}}

/* 📝 Input fields */
.stTextInput>div>div>input,
.stTextArea>div>div>textarea {{
    border-radius: 12px !important;
    padding: 12px !important;
    border: 1px solid {current_theme['border']} !important;
    background-color: {current_theme['input_bg']} !important;
    color: {current_theme['input_text']} !important;
}}

/* 🔍 Placeholder */
.stTextInput>div>div>input::placeholder,
.stTextArea>div>div>textarea::placeholder {{
    color: {current_theme['input_text']} !important;
    opacity: 0.7 !important;
}}

/* 🎯 Focus effect */
.stTextInput>div>div>input:focus,
.stTextArea>div>div>textarea:focus {{
    border-color: {current_theme['primary']} !important;
    box-shadow: 0 0 0 2px {current_theme['primary']}33 !important;
}}

/* 🧾 Custom card */
.custom-card {{
    background-color: {current_theme['card']} !important;
    border-radius: 16px !important;
    padding: 20px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    color: {current_theme['text']} !important;
}}

/* ✅ Confidence indicators */
.confidence-high {{ color: #28a745; font-weight: bold; }}
.confidence-medium {{ color: #ffc107; font-weight: bold; }}
.confidence-low {{ color: #dc3545; font-weight: bold; }}

/* 👋 Greeting */
.greeting {{
    font-size: 1.2rem;
    padding: 10px 0px 0px 0px;
    margin-bottom: 20px;
    text-align: center;
    color: {current_theme['text']} !important;
}}

/* 🔧 Tool headers */
.tool-header {{
    font-size: 1.1rem;
    margin-bottom: 15px;
    color: {current_theme['text']} !important;
    font-weight: 600;
}}

/* 📑 Document list */
.document-list {{
    padding: 10px;
    background-color: {current_theme['card']} !important;
    border-radius: 8px !important;
    margin-bottom: 15px !important;
}}

/* 📤 File uploader box */
.stFileUploader {{
    background-color: {current_theme['card']} !important;
    padding: 15px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05) !important;
}}

/* ⬆️ Upload button */
.stFileUploader>div>div>div>button {{
    background-color: {current_theme['secondary']} !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 10px 20px !important;
    border: none !important;
    transition: all 0.3s ease;
}}

/* 🔘 Secondary buttons */
.stButton>button[kind="secondary"] {{
    background-color: {current_theme['secondary']} !important;
    color: {current_theme['button_text']} !important;
}}
</style>
""", unsafe_allow_html=True)


# -------------------- Brand Logo in Header --------------------
header_logo_path = "images/ScholarMate.png"  # Update this path to your actual logo file

try:
    # Read and encode the logo image
    with open(header_logo_path, "rb") as image_file:
        encoded_logo = base64.b64encode(image_file.read()).decode()
    
    # Replace the gradient header with logo version
    st.markdown(f"""
    <style>        
        .header-logo {{
            height: 150px;
            width: auto;
            object-fit: cover;
        }}
    </style>

    <div class="gradient-header">
        <img class="header-logo" src="data:image/png;base64,{encoded_logo}">
    </div>
    """, unsafe_allow_html=True)

except FileNotFoundError:
    # Fallback to text version if logo not found
    st.markdown(f"""
    <div class="gradient-header">
        <h1 style="color:white; margin-bottom:10px;">ScholarMate</h1>
        <p style="color:white; font-size:16px;">Transform your documents into interactive learning resources</p>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"Error loading logo: {str(e)}")

# -------------------- Sidebar --------------------
with st.sidebar:
    # Logo and header
    logo_path = "images/logo.png"
    st.markdown(f"""
<div class="brand-logo">
    <img src="data:image/png;base64,{base64.b64encode(open(logo_path, "rb").read()).decode()}">
    """, unsafe_allow_html=True)
    
    # Theme Selector
    with st.expander("🎨 Theme Settings", expanded=True):
        theme_options = ['academic', 'ocean', 'sunset', 'forest', 'royal']
        st.session_state.theme = st.selectbox(
            "Select Theme",
            theme_options,
            index=theme_options.index(st.session_state.theme))
        st.session_state.dark_mode = st.toggle("Dark Mode", value=st.session_state.dark_mode)
        
        if st.button("Apply Theme Changes"):
            st.rerun()
    
    # Model Settings
    with st.expander("⚙️ Settings", expanded=True):
        temperature = st.slider("Creativity", 0.0, 1.0, 0.3, 0.1, 
                              help="Lower for factual answers, higher for creative responses")
        chunk_size = st.slider("Text Chunk Size", 500, 2000, 1000, 100, 
                             help="Size of document chunks for processing")
    
    # Instructions
    with st.expander("📚 Instructions", expanded=True):
        st.markdown("""
        1. **Upload** your academic documents (PDF, DOCX, TXT)
        2. **Ask questions** about the content
        3. Use the **tools** to generate summaries, MCQs, and explanations
        """)

# Initialize LLM
llm = ChatGroq(model_name="llama3-70b-8192", temperature=temperature)

# -------------------- Document Processing Functions --------------------
def load_documents(file_objects):
    """Load and process uploaded documents, keeping only content"""
    documents = []
    for file in file_objects:
        file_name = file.name.lower()
        temp_path = f"temp_{file_name}"
        
        try:
            with open(temp_path, "wb") as f:
                f.write(file.read())

            if file_name.endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
            elif file_name.endswith(".docx"):
                loader = Docx2txtLoader(temp_path)
            elif file_name.endswith(".txt"):
                loader = TextLoader(temp_path)
            else:
                continue

            loaded_docs = loader.load()
            for doc in loaded_docs:
                # Create a proper Document object instead of dictionary
                documents.append(Document(
                    page_content=doc.page_content,
                    metadata=doc.metadata  # Preserve metadata if needed
                ))
                
        except Exception as e:
            st.error(f"Error processing {file_name}: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    return documents

def create_retriever(docs):
    """Create vector store and retriever from document content"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        length_function=len
    )
    
    # No need to convert to Document objects here since they're already Documents
    chunks = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})

# -------------------- RAG Pipeline Functions --------------------
def run_rag_chain(question_text, retriever_obj, prompt_template):
    """Execute the RAG pipeline with metadata filtering"""
    def clean_context(docs):
        return [doc.page_content.replace("Title:", "").replace("Source:", "").strip() 
                for doc in docs]
    
    prompt = PromptTemplate.from_template(prompt_template)
    rag_chain = (
        RunnableParallel({
            "context": retriever_obj | clean_context,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question_text)

def calculate_confidence_score(answer: str, context_chunks: list):
    """Calculate confidence score for answers"""
    embedder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    answer_embedding = embedder.embed_query(answer)
    context_text = " ".join([chunk.page_content for chunk in context_chunks])
    context_embedding = embedder.embed_query(context_text)

    score = cosine_similarity(
        [np.array(answer_embedding)],
        [np.array(context_embedding)]
    )[0][0]

    return round(float(score), 2)

# -------------------- Tool Definitions --------------------
@tool
def summarize_tool(tool_input: str = "") -> str:
    """Summarizes the uploaded academic documents."""
    if not st.session_state.get("docs"):
        return "Please upload and process documents first"
    
    prompt = """You are an academic assistant. Create a comprehensive yet concise summary of the documents. 
Only use the actual textual content - ignore any metadata, file names, or document structure information.
Organize the summary with clear sections and bullet points where appropriate. 

Context (textual content only):
{context}

Important: Do not include any:
- File names or paths
- Page numbers
- Document source information
- Formatting metadata

Summary:
"""
    return run_rag_chain("Summarize the uploaded documents.", st.session_state["retriever"], prompt)

@tool
def mcq_tool(tool_input: str = "") -> str:
    """Generates multiple-choice questions based on uploaded documents."""
    if not st.session_state.get("docs"):
        return "Please upload and process documents first"
    
    prompt = """You are an academic assistant. Generate 5 high-quality multiple-choice questions (MCQs) based on the textual content only.
Ignore any metadata or document structure information.

For each question:
- Provide 4 plausible options (A-D)
- Mark the correct answer with (Correct)
- Include brief explanations for the correct answers
- Organize by difficulty level (Basic, Intermediate, Advanced)

Context (textual content only):
{context}

Important: Base questions only on the actual content, not on:
- File names
- Page numbers
- Source information
- Any document metadata

MCQs:
"""
    return run_rag_chain("Generate 5 MCQs from the documents.", st.session_state["retriever"], prompt)

@tool
def explain_tool(tool_input: str) -> str:
    """Provides a detailed explanation of topics based on uploaded documents."""
    if not st.session_state.get("docs"):
        return "Please upload and process documents first"
    if not tool_input:
        return "Please provide a topic to explain"
    
    prompt = """You are an academic expert. Provide a detailed explanation of the topic "{question}" using only the textual content below.
Completely ignore any metadata or document structure information.

Structure your response with:
1. Definition and key concepts (from content only)
2. Theoretical background (from content only)
3. Practical applications (if applicable)
4. Relation to other concepts in the documents

Context (textual content only):
{context}

Important: Do not reference:
- Document titles
- Page numbers
- File names
- Source information
- Any non-content metadata

Detailed Explanation:
"""
    return run_rag_chain(tool_input, st.session_state["retriever"], prompt)

# -------------------- Main Content --------------------
st.markdown(f"""
<div class="greeting">
    {get_greeting()}
</div>
""", unsafe_allow_html=True)

# -------------------- File Upload Section --------------------
with st.expander("📤 Upload Documents", expanded=True):
    uploaded_files = st.file_uploader(
        "Upload your academic files (PDF, DOCX, TXT):",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files and uploaded_files != st.session_state.get("uploaded_files"):
        st.session_state.uploaded_files = uploaded_files
        
        with st.spinner("🔍 Processing documents..."):
            try:
                docs = load_documents(uploaded_files)
                if docs:
                    retriever = create_retriever(docs)
                    st.session_state.docs = docs
                    st.session_state.retriever = retriever
                    st.success(f"✅ Processed {len(uploaded_files)} file(s)")
                else:
                    st.error("No valid content found in the uploaded files")
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")

    if st.session_state.get("uploaded_files"):
        st.markdown(f"""
        <div class="custom-card">
            <h3>Uploaded Documents</h3>
            <div class="document-list">
                {''.join([f'<p>• {file.name}</p>' for file in st.session_state.uploaded_files])}
            </div>
        </div>
        """, unsafe_allow_html=True)

# -------------------- Question Answering Section --------------------
with st.container():
    st.markdown(f"""
    <div class="custom-card">
        <h2>🔍 Ask Your Question</h2>
    </div>
    """, unsafe_allow_html=True)

    question = st.text_area(
        "Type your academic question here:",
        placeholder="e.g., Explain the key concepts of quantum mechanics as discussed in the documents...",
        height=120,
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([4, 1])
    with col1:
        if st.button("Get Answer", use_container_width=True):
            if not st.session_state.get("uploaded_files") or not question:
                st.warning("Please upload at least one document and enter a question.")
            elif not st.session_state.get("retriever"):
                st.warning("Documents are still processing. Please wait...")
            else:
                with st.spinner("🧠 Analyzing documents..."):
                    try:
                        answer_prompt = """You are a precise academic assistant. Answer the question using ONLY the context below.
Ignore any metadata, formatting information, or document structure tags. Focus solely on the textual content.
Maintain an academic tone and provide clear, well-structured explanations. 

Context:
{context}

Question: {question}

Important: Only use the actual textual content from the documents. Do not reference any metadata like:
- File names
- Page numbers
- Document titles
- Source information
- Formatting tags

Provide a detailed answer based solely on the textual content:
"""
                        answer = run_rag_chain(question, st.session_state.retriever, answer_prompt)
                        confidence = calculate_confidence_score(answer, st.session_state.docs)

                        st.markdown(f"""
                        <div class="custom-card">
                            <h3 style="border-bottom:1px solid {current_theme['border']}; padding-bottom:10px;">Answer</h3>
                            <div style="font-size:16px; line-height:1.6;">
                                {answer}
                            </div>
                            <div style="margin-top:20px; padding-top:10px; border-top:1px solid {current_theme['border']};">
                                <span style="font-weight:bold;">Confidence:</span> 
                                <span class="{'confidence-high' if confidence > 0.7 else 'confidence-medium' if confidence > 0.4 else 'confidence-low'}">
                                    {confidence} ({"High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"})
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")

    with col2:
        if st.button("Clear All", type="secondary", use_container_width=True):
            st.session_state.clear()
            st.rerun()

# -------------------- Academic Tools Section --------------------
st.markdown(f"""
<div class="custom-card">
    <h2>🛠️ Academic Tools</h2>
    <p>Enhance your learning with these powerful tools</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📝 Summarize", "❓ Generate MCQs", "📚 Topic Explanation"])

with tab1:
    with st.container():
        st.markdown(f"""
        <div class="tool-header">
            Document Summary Generator
        </div>
        <p>Create concise summaries of your uploaded materials</p>
        """, unsafe_allow_html=True)
        
        if st.button("Generate Summary", key="summary_btn", use_container_width=True):
            if not st.session_state.get("docs"):
                st.warning("Please upload and process documents first")
            else:
                with st.spinner("📝 Generating summary..."):
                    try:
                        result = summarize_tool("")
                        st.session_state.summary_output = result
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
        
        if st.session_state.get("summary_output"):
            st.markdown(f"""
            <div class="custom-card">
                <div style="font-size:16px; line-height:1.6;">
                    {st.session_state.summary_output}
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    with st.container():
        st.markdown(f"""
        <div class="tool-header">
            MCQ Generator
        </div>
        <p>Create practice questions for self-assessment</p>
        """, unsafe_allow_html=True)
        
        if st.button("Generate MCQs", key="mcq_btn", use_container_width=True):
            if not st.session_state.get("docs"):
                st.warning("Please upload and process documents first")
            else:
                with st.spinner("📝 Generating practice questions..."):
                    try:
                        result = mcq_tool("")
                        st.session_state.mcqs_output = result
                    except Exception as e:
                        st.error(f"Error generating MCQs: {str(e)}")
        
        if st.session_state.get("mcqs_output"):
            st.markdown(f"""
            <div class="custom-card">
                <div style="font-size:16px; line-height:1.6;">
                    {st.session_state.mcqs_output}
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    with st.container():
        st.markdown(f"""
        <div class="tool-header">
            Topic Explainer
        </div>
        <p>Get detailed explanations of specific concepts</p>
        """, unsafe_allow_html=True)
        
        topic_input = st.text_input(
            "Enter a topic to explain:",
            placeholder="e.g., Quantum entanglement, Neural networks, French Revolution...",
            key="topic_input",
            label_visibility="collapsed"
        )
        
        if st.button("Explain Topic", key="explain_btn", use_container_width=True):
            if not topic_input:
                st.warning("Please enter a topic first")
            elif not st.session_state.get("docs"):
                st.warning("Please upload and process documents first")
            else:
                with st.spinner(f"📚 Generating explanation..."):
                    try:
                        result = explain_tool(topic_input)
                        st.session_state.topic_explanation_output = result
                    except Exception as e:
                        st.error(f"Error generating explanation: {str(e)}")
        
        if st.session_state.get("topic_explanation_output"):
            st.markdown(f"""
            <div class="custom-card">
                <div style="font-size:16px; line-height:1.6;">
                    {st.session_state.topic_explanation_output}
                </div>
            </div>
            """, unsafe_allow_html=True)

# -------------------- Footer --------------------
st.markdown("""
<div style="margin-top:50px; padding:20px; text-align:center; color:#6c757d; border-top:1px solid #eee;">
    <p>ScholarMate • For educational purposes only</p>
</div>
""", unsafe_allow_html=True)