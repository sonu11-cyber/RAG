
import streamlit as st 
#import os
#from dotenv import load_dotenv
#Langchain Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

#Page conf
st.set_page_config(page_title='C++ RAG Chatbot')
st.title('C++ RAG Chatbot')
st.write('Ask any question related to C++ iNTRODUCTION')

#load env varfiables


@st.cache_resource
def load_vectorstore():
    loader = TextLoader('C++ Introduction.txt',encoding='utf-8')
    documents=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    final_documents=text_splitter.split_documents(documents)

    #embeddings
    Embeddings= HuggingFaceEmbeddings(
       model_name='all-MiniLM-L6-v2'
       )

   

    db= FAISS.from_documents(final_documents,Embeddings)
    return db

db=load_vectorstore()
llm= Ollama(model='gemma2:2b')
user_question=st.text_input('ask a question')

if user_question:
    with st.spinner('Thinking...'):
        docs=db.similarity_search(user_question)
        context='\n'.join([doc.page_content for doc in docs])
        prompt = f"""
        Answer the question using only the context below
        

        Context:
        {context}

        Question:
        {user_question}

        Answer:
        """

        response=llm.invoke(prompt)
        st.subheader('ANSWER:')
        st.write(response)