# Databricks notebook source
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter            ## convert into chunks
from langchain.chains.combine_documents import create_stuff_documents_chain   ## helps in fetching relevant docs in q&a
from langchain_core.prompts import ChatPromptTemplate                         ## create custom prompt
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS                            ## it is a vector DB to store vectors
from langchain_community.document_loaders import PyPDFDirectoryLoader         
from langchain_google_genai import GoogleGenerativeAIEmbeddings               #converts chunks into vectors/embeddings
from dotenv import load_dotenv

#set .env file
load_dotenv()

#load GROQ and GOOGLE API from .env
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

#title of the app
st.title("Mini-GPT")

#import llm model using groq api key
llm = ChatGroq(groq_api_key = groq_api_key, model_name = "Gemma-7b-it")


#set prompt template
prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

#function to read pdf, convert into chunks, embbed and store in vectordb FAISS 

def vector_embedding():

    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./data_pdfs") ## Data Ingestion
        st.session_state.docs = st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings


prompt1=st.text_input("sample test")


if st.button("VectorDB"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time

if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
