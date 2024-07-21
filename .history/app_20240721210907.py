from dotenv import load_dotenv
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

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

def get_vector_store(text_chunks, api_key):
    # Ensure the correct model name format for embeddings
    model_name = "models/embedding-001"  # Verify this model name in the documentation
    print(f"Using model for embeddings: {model_name}")
    embeddings = GoogleGenerativeAIEmbeddings(model=model_name, api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(api_key):
    default_prompt = """Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in the provided context, just say "answer is not available in the context". Don't provide the wrong answer.\n\n
    Context: \n{context}\n
    Question: \n{question}\n
    
    Answer: 
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, api_key=api_key)
    prompt = PromptTemplate(template=default_prompt, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    model_name = "models/embedding-001"  # Ensure this is correct
    embeddings = GoogleGenerativeAIEmbeddings(model=model_name, api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain(api_key)
    
    response = chain(
        {
            "input_documents": docs, 
            "question": user_question
        },
        return_only_outputs=True
    )
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chat with multiple PDF")
    st.header("Chat with multiple PDF")

    api_key = st.text_input("Enter your personal API key:")
submit_api = st.button("Submit Key")

if submit_api and not api_key:
    st.warning("Please enter your personal API key to proceed.")
    st.stop()
    
    if submit_key:
        genai.configure(api_key=api_key)
        
        user_question = st.text_input("Ask a Question from the PDF Files")
        submit_question = st.button("Ask")
        
        if submit_question:
            user_input(user_question, api_key)
            
        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your PDF Files", type=["pdf"], accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, api_key)
                    st.success("Done")

if __name__ == "__main__":
    main()
