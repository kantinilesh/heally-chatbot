import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
import os
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize company knowledge base
def initialize_company_knowledge():
    # Load and process company documents
    company_data = """
    Collabor8 is an educational technology platform focused on improving classroom connectivity
    and reducing inequity in education. The platform helps connect students, professors, and
    teachers while addressing issues of disconnection, inequity, and distraction in modern
    education.
    
    Key Features:
    - Group studying and office hours support
    - Career help and mentorship
    - Volunteer opportunities for college students
    - Tools for professors to organize group assignments
    - High school student guidance
    - Industry professional connections
    
    Pricing:
    - Individual students: $8.95/month
    - Colleges: $2.95/student/month
    - High schools: Free
    """
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    company_texts = text_splitter.split_text(company_data)
    company_db = FAISS.from_texts(company_texts, embeddings)
    return company_db

# Function to process uploaded PDF
def process_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(pages)
    
    db = FAISS.from_documents(texts, embeddings)
    os.unlink(tmp_path)
    return db

# Initialize Streamlit app
def main():
    st.title("Heally Assistant")
    st.write("Hello! I'm your Heally assistant. I can help you with questions about Collabor8 and analyze any PDF documents you upload.")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'company_db' not in st.session_state:
        st.session_state.company_db = initialize_company_knowledge()
    if 'pdf_db' not in st.session_state:
        st.session_state.pdf_db = None
        
    # Common questions
    st.sidebar.header("Common Questions")
    if st.sidebar.button("What is Collabor8?"):
        question = "What is Collabor8?"
    elif st.sidebar.button("What are the pricing plans?"):
        question = "What are the pricing plans?"
    elif st.sidebar.button("What features do you offer?"):
        question = "What features do you offer?"
    else:
        question = None
        
    # PDF upload
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=['pdf'])
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            st.session_state.pdf_db = process_pdf(uploaded_file)
        st.success("PDF processed successfully!")
        
    # Custom question input
    custom_question = st.text_input("Ask your question:")
    if custom_question:
        question = custom_question
        
    # Process question if exists
    if question:
        try:
            # Initialize LLM and chain
            llm = HuggingFaceHub(
                repo_id="google/flan-t5-large",
                huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
            )
            
            # Combine knowledge bases if PDF is uploaded
            if st.session_state.pdf_db:
                company_docs = st.session_state.company_db.similarity_search(question)
                pdf_docs = st.session_state.pdf_db.similarity_search(question)
                all_docs = company_docs + pdf_docs
                retriever = lambda q: all_docs
            else:
                retriever = st.session_state.company_db.as_retriever()
                
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )
            
            # Get response
            response = chain({"question": question, "chat_history": st.session_state.chat_history})
            
            # Update chat history
            st.session_state.chat_history.append((question, response['answer']))
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
        # Display chat history
        for q, a in st.session_state.chat_history:
            st.write(f"Q: {q}")
            st.write(f"A: {a}")
            st.write("---")

if __name__ == "__main__":
    main()
