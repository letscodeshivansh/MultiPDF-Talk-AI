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

def get_vector_store(text_chunks, api_key):
    # Ensure the correct model name format for embeddings
    model_name = "models/embedding-001"  # Verify this model name in the documentation
    embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(api_key):
    default_prompt = """Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in the provided context, just say "answer is not available in the context". Don't provide the wrong answer.\n\n
    Context: \n{context}\n
    Question: \n{question}\n
    
    Answer: 
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=default_prompt, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    model_name = "models/embedding-001"  # Ensure this is correct
    embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
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



if __name__ == "__main__":
    main()
