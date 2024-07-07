import streamlit as st
import ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

embedding_fn = OllamaEmbeddings(model='llama3')
vectorstore = Chroma(persist_directory='./data/vectorstore', embedding_function=embedding_fn)
retriever = vectorstore.as_retriever()

def infer(prompt):
    req = [{'role': 'user', 'content': prompt}]
    resp = ollama.chat(model='llama3', messages=req)
    return resp['message']['content']

def infer_with_context(question):
    docs = retriever.invoke(question)
    ctx = '\n\n'.join(doc.page_content for doc in docs)
    prompt = f"""
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

        Question: {question} 

        Context: {ctx} 

        Answer:
    """
    return infer(prompt)

st.title('testy testerino')
st.caption('nyahallo')

question = st.text_input('Chat with Ragdoll')
if question:
    answer = infer_with_context(question)
    st.write(answer)