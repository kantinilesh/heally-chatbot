import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()  # Load variables from .env

# Initialize embeddings
def get_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        return embeddings
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return None

# Initialize embeddings model
embeddings_model = get_embeddings()

# Initialize company knowledge base
def initialize_company_knowledge():
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
    
    try:
        if embeddings_model is None:
            st.error("Embeddings model not initialized")
            return None
        
        company_db = FAISS.from_texts(company_texts, embeddings_model)
        return company_db
    except Exception as e:
        st.error(f"Error creating company knowledge base: {e}")
        return None

# Function to process uploaded PDF
def process_pdf(pdf_file):
    try:
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
        
        if embeddings_model is None:
            st.error("Embeddings model not initialized")
            return None
        
        db = FAISS.from_documents(texts, embeddings_model)
        os.unlink(tmp_path)
        return db
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

def main():
    st.title("Heally Assistant")
    st.write("Hello! I'm your Heally assistant. I can help you with questions about Collabor8 and analyze PDF documents.")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize company knowledge base
    try:
        if 'company_db' not in st.session_state:
            st.session_state.company_db = initialize_company_knowledge()
            if st.session_state.company_db is None:
                st.error("Failed to initialize company knowledge base")
    except Exception as e:
        st.error(f"Failed to initialize company knowledge base: {e}")
    
    if 'pdf_db' not in st.session_state:
        st.session_state.pdf_db = None
        
    # Common questions
    common_questions = {
        "What is Collabor8?": "Provide an overview of Collabor8",
        "Pricing Details": "Explain the pricing plans",
        "Key Features": "List the main features of the platform"
    }
    
    st.sidebar.header("Common Questions")
    selected_question = st.sidebar.radio("Select a common question:", list(common_questions.keys()))
    
    # PDF upload
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=['pdf'])
    if uploaded_file:
        try:
            with st.spinner("Processing PDF..."):
                st.session_state.pdf_db = process_pdf(uploaded_file)
            st.success("PDF processed successfully!")
        except Exception as e:
            st.error(f"PDF upload error: {e}")
    
    # Question input
    question = st.text_input("Ask your question:", value=common_questions[selected_question])
    
    if st.button("Get Answer"):
        try:
            # Ensure company_db is not None
            if st.session_state.company_db is None:
                st.error("Company knowledge base not initialized")
                return
            
            # Use a local LLM 
            llm = HuggingFaceHub(
                repo_id="google/flan-t5-small",
                huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
            )
            
            # Create retrievers
            company_retriever = st.session_state.company_db.as_retriever(search_kwargs={"k": 2})
            
            # Retrieve context
            if st.session_state.pdf_db:
                pdf_retriever = st.session_state.pdf_db.as_retriever(search_kwargs={"k": 2})
                # Combine retrievers
                from langchain.retrievers import BM25Retriever, EnsembleRetriever
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[company_retriever, pdf_retriever],
                    weights=[0.5, 0.5]
                )
                retriever = ensemble_retriever
            else:
                retriever = company_retriever
            
            # Generate response
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )
            
            response = chain({"question": question, "chat_history": st.session_state.chat_history})
            
            # Display response
            st.write("Answer:", response['answer'])
            
            # Display source documents
            if response.get('source_documents'):
                st.write("Sources:")
                for doc in response['source_documents']:
                    st.write(doc.page_content[:300] + "...")
            
            # Update chat history
            st.session_state.chat_history.append((question, response['answer']))
        
        except Exception as e:
            st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()