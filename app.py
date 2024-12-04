import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            try:
                text += page.extract_text()
            except Exception as e:
                st.error(f"Error extracting text from page: {e}")
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save the vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create the conversational chain
def get_conversational_chain():
    # Define the prompt template
    prompt = PromptTemplate(
        template="""
        Answer the question as detailed as possible from the provided context. 
        If the answer is not available in the context, respond with 
        "The answer is not available in the context." Do not provide incorrect information.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """,
        input_variables=["context", "question"],  # Correctly specify input variables
    )
    # Initialize the model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    # Load the question-answering chain with the prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and generate responses
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Load the FAISS index with deserialization
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # Perform similarity search
    docs = new_db.similarity_search(user_question)

    # Get the conversational chain
    chain = get_conversational_chain()

    # Pass inputs to the chain
    response = chain(
        {"input_documents": docs, "question": user_question},  # Corrected key 'question'
        return_only_outputs=True
    )

    st.write("Reply:", response.get("output_text", "No response generated."))

# Main Streamlit app
def main():
    st.set_page_config(page_title="Chat with Multiple PDFs")
    st.header('Chat with PDFs using Gemini')
    
    # Sidebar for file upload
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)
        if st.button("Submit and Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete. You can now ask questions!")
            else:
                st.warning("Please upload PDF files before clicking submit.")

    # Main input and response display
    user_question = st.text_input('Ask a question based on the uploaded PDFs:')
    if user_question.strip():
        user_input(user_question)
    else:
        st.warning("Please enter a valid question.")

# Run the app
if __name__ == "__main__":
    main()
