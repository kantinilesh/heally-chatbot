import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import os
import tempfile

def get_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        return embeddings
    except Exception as e:
        st.error(f"Embeddings error: {e}")
        return None

def initialize_llm():
    try:
        # Use OpenAI's ChatGPT model
        llm = ChatOpenAI(
            openai_api_key="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",
            model_name="gpt-3.5-turbo",  # You can change to gpt-4 if you prefer
            temperature=0.1,
            max_tokens=200
        )
        return llm
    except Exception as e:
        st.error(f"LLM initialization error: {e}")
        return None

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
        embeddings = get_embeddings()
        if embeddings is None:
            st.error("Embeddings not available")
            return None
        
        company_db = FAISS.from_texts(company_texts, embeddings)
        return company_db
    except Exception as e:
        st.error(f"Company knowledge base error: {e}")
        return None

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
        
        embeddings = get_embeddings()
        if embeddings is None:
            st.error("Embeddings not available")
            return None
        
        db = FAISS.from_documents(texts, embeddings)
        os.unlink(tmp_path)
        return db
    except Exception as e:
        st.error(f"PDF processing error: {e}")
        return None

def main():
    st.title("Collabor8 Assistant")
    st.write("Hello! I'm your Collabor8 assistant. I can help you with questions about the platform and analyze PDF documents.")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    try:
        if 'company_db' not in st.session_state:
            st.session_state.company_db = initialize_company_knowledge()
        
        if 'llm' not in st.session_state:
            st.session_state.llm = initialize_llm()
    except Exception as e:
        st.error(f"Initialization error: {e}")
    
    if 'pdf_db' not in st.session_state:
        st.session_state.pdf_db = None
        
    common_questions = {
        "What is Collabor8?": "Provide an overview of Collabor8",
        "Pricing Details": "Explain the pricing plans",
        "Key Features": "List the main features of the platform"
    }
    
    st.sidebar.header("Common Questions")
    selected_question = st.sidebar.radio("Select a common question:", list(common_questions.keys()))
    
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=['pdf'])
    if uploaded_file:
        try:
            with st.spinner("Processing PDF..."):
                st.session_state.pdf_db = process_pdf(uploaded_file)
            st.success("PDF processed successfully!")
        except Exception as e:
            st.error(f"PDF upload error: {e}")
    
    question = st.text_input("Ask your question:", value=common_questions[selected_question])
    
    if st.button("Get Answer"):
        try:
            if st.session_state.company_db is None or st.session_state.llm is None:
                st.error("Systems not fully initialized")
                return
            
            company_retriever = st.session_state.company_db.as_retriever(search_kwargs={"k": 2})
            
            if st.session_state.pdf_db:
                pdf_retriever = st.session_state.pdf_db.as_retriever(search_kwargs={"k": 2})
                from langchain.retrievers import EnsembleRetriever
                retriever = EnsembleRetriever(
                    retrievers=[company_retriever, pdf_retriever],
                    weights=[0.5, 0.5]
                )
            else:
                retriever = company_retriever
            
            chain = ConversationalRetrievalChain.from_llm(
                llm=st.session_state.llm,
                retriever=retriever,
                return_source_documents=True
            )
            
            response = chain({"question": question, "chat_history": st.session_state.chat_history})
            
            st.write("Answer:", response['answer'])
            
            if response.get('source_documents'):
                st.write("Sources:")
                for doc in response['source_documents']:
                    st.write(doc.page_content[:300] + "...")
            
            st.session_state.chat_history.append((question, response['answer']))
        
        except Exception as e:
            st.error(f"Response generation error: {e}")

if __name__ == "__main__":
    main()
