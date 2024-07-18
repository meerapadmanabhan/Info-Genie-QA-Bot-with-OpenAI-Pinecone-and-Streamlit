# infogenie.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
import openai
import pinecone
import numpy as np
from langchain_community.retrievers import PineconeHybridSearchRetriever

# Initialize Pinecone
api_key = "7f1a3949-c165-400d-9f11-6b2e1ae4a9c0"
index_name = "infogenie"

# Initialize Pinecone client
pinecone.init(api_key=api_key)

# Create index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=384,  # Dimensionality of dense model
        metric="dotproduct",
        spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pinecone.Index(index_name)

openai.api_key = "sk-None-xwNqvKAZBN9x6HfllXT9T3BlbkFJfniQeZb7ezjy9ZO8Mn6U"  # Use environment variable or secret in production

def fetch_website_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    content = "\n".join([para.get_text() for para in paragraphs])
    return content

def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    embedding = response['data'][0]['embedding']
    return np.array(embedding)

def insert_website_content(url):
    content = fetch_website_content(url)
    paragraphs = content.split('\n')
    for i, paragraph in enumerate(paragraphs):
        if paragraph.strip():
            embedding = get_embedding(paragraph)
            index.upsert([(f"{url}_para_{i}", embedding)])

def query_pinecone(query):
    query_embedding = get_embedding(query)
    results = index.query(query_embedding, top_k=5, include_values=True)
    return results

def generate_answer(query, context):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Answer the following question based on the context:\n\nContext: {context}\n\nQuestion: {query}",
        max_tokens=200
    )
    return response.choices[0].text.strip()

def qa_bot(query):
    results = query_pinecone(query)
    if results['matches']:
        context = " ".join([res['value'] for res in results['matches']])
        answer = generate_answer(query, context)
    else:
        answer = "No relevant information found in the indexed content."
    return answer

st.title("InfoGenie🤖")

url = st.text_input("Enter the website URL:")
query = st.text_input("Enter your question:")

if st.button("Fetch and Index Content"):
    with st.spinner("Fetching and indexing content..."):
        insert_website_content(url)
        st.success("Content indexed successfully!")

if st.button("Get Answer"):
    with st.spinner("Fetching answer..."):
        answer = qa_bot(query)
        st.write("Answer:", answer)


