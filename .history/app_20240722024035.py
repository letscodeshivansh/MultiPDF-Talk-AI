from dotenv import load_dotenv
import streamlit as st
import os
import logging
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the environment variables
load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    try:
        # Ensure the correct model name format for embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        raise

def get_conversational_chain():
    default_prompt = """Answer the question as detailed as possible from the provided context. Make sure to provide all the details. Don't provide the wrong answer.\n\n
    Context: \n{context}\n
    Question: \n{question}\n
    
    Answer: 
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=default_prompt, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        
        response = chain(
            {
                "input_documents": docs, 
                "question": user_question
            },
            return_only_outputs=True
        )
        st.write("Reply: ", response["output_text"])
    except Exception as e:
        logging.error(f"Error processing user input: {e}")
        st.error(f"An error occurred: {e}")

def main():
    st.set_page_config(page_title="Chat with multiple PDF")
    st.header("Chat with multiple PDF")

    api_key = st.text_input("Enter your personal API key:", type="password")
    submit_api = st.button("Submit Key")

    if submit_api and not api_key:
        st.warning("Please enter your personal API key to proceed.")
        st.stop()
    
    if api_key:
        genai.configure(api_key=api_key)
        
        user_question = st.text_input("Ask a Question from the PDF Files")
        submit_question = st.button("Ask")
        
        if submit_question:
            user_input(user_question)
            
        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your PDF Files", type=["pdf"], accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done")
                    except Exception as e:
                        logging.error(f"Error processing PDF files: {e}")
                        st.error(f"An error occurred while processing the files: {e}")

if __name__ == "__main__":
    main()
