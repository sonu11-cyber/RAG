
from langchain_community.llms import Ollama
import streamlit as str 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#crreating my orompts 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",'You are helpful assistant.please respond to the question asked '),
        ("user","Question:{question}")
    ]
)

#streamlit framework
str.title('Langchain demo with gemma2:2b')
input_text=str.text_input('What question do u have in your mind')

#Ollama model
llm=Ollama(model="gemma2:2b")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    str.write(chain.invoke({"question":input_text}))
