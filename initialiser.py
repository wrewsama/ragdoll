# script to encode and store the markdown data into a chroma sqlite

import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

loader = DirectoryLoader('./knowledge_base', '**/*.md')
docs = loader.load()
print(docs)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
splits = splitter.split_documents(docs)

print('pulling model')
ollama.pull('llama3')
print('model pulled')
embedding_fn = OllamaEmbeddings(model='llama3')
print('creating vectors')
Chroma.from_documents(documents=splits, embedding=embedding_fn , persist_directory='./data/vectorstore')
