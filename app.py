import time
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
load_dotenv()

# Set API keys from environment
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name='Llama3-8b-8192')

# Define the prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    Adhere to these instructions AS STRICTLY AS POSSIBLE when answering the question.
    - Answer the questions as if you are ABHINAV BANDARU.
    - Completely Refrain from using phrases like "Based on the provided context", "From the document", "According to Abhinav Bandaru", or anything in that capacity
    - Answer the questions based on the provided context about Abhinav Bandaru. Use the document to get more information about Abhinav.
    - DO NOT ASSUME ANYTHING about Abhinav Bandaru, other than what's given. If something is obviously inferrable, only then are you allowed to make very obvious assumptions.
    - Keep the answers brief, concise and to the point. Don't repeat yourself. Make sure you get the point across in the answer.
    - If there's a question that you do not know the answer to, direct the user to forward the queries to the Human version of Abhinav Bandaru by contacting him at abhinow@seas.upenn.edu
    - Any questions about Work Experience/Projects, make sure to talk briefly about NASA and Fox Entertainment first. If more projects/work experience is required only then dive into the other experiences
    - Answer any question about your identity along the lines (but not as it is) of, 'Heyy there! I'm Abhinav, a candidate for Master's in Data Science at the University of Pennsylvania. I'm set to graduate in summmer of 2025 and am looking for Machine Learning & Data Science Roles starting June 2025.'
    
    Context:
    {context}
    
    Question:{input}
    """
)

# Session state initialization
if 'awake' not in st.session_state:
    st.session_state.awake = False

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader(
            "About Me")  # Data Ingestion Step
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings)


# Set the title of the app
title = f'Heyy! This is Abhinav! What can I do for you?'
st.title(title)

# Button to wake up the app and create the vector embedding
if st.button("Click Me to Wake me up!"):
    create_vector_embedding()
    st.session_state.awake = True
    st.write('Vector Database is ready!')


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


# Only display input field if the app is awake
if st.session_state.awake:
    # Text input for user prompt
    user_prompt = st.text_input("What's up?", key="user_input")

    # Check if there's any input from the user
    if user_prompt:
        prompt = prompt_template

        # Create the document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()

        # Retrieve similar documents based on the user prompt
        similar_docs = retriever.get_relevant_documents(user_prompt)

        # Format the retrieved documents into a context string
        context = format_docs(similar_docs)

        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Timing the response generation
        start = time.process_time()
        response = retrieval_chain.invoke({
            'input': user_prompt,
            'context': context,
        })
        response_time = time.process_time() - start

        # Display the response and response time
        st.write(f"Response time: {response_time:.2f} seconds")
        st.write(response['answer'])

        # Display documents used for similarity search in an expander
        with st.expander("Document similarity search"):
            for i, doc in enumerate(similar_docs):
                st.write(doc.page_content)
                st.write('---------------')
