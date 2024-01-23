from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams
from genai.model import Credentials
import os
import PyPDF2
import random
import itertools
import streamlit as st
from io import StringIO
from langchain.chains import RetrievalQA
from langchain.retrievers import SVMRetriever
from langchain.chains import QAGenerationChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


# Page title
st.title('Context aware Retriever Augmented Generation Demo powered by IBM Watsonx')
st.caption("This demo is prepared by Sharath Kumar RK, Senior Data Scientist, IBM Watsonx team")
st.subheader("Ask questions about your document")


genai_api_url = st.sidebar.text_input("GenAI API URL", type="password", value="https://bam-api.res.ibm.com/v1")
model = 'meta-llama/llama-2-70b-chat'
chunk_size = st.sidebar.number_input("Select chunk size", value=1000)
chunk_overlap = st.sidebar.number_input("Select chunk overlap", value=0)
maximum_new_tokens = st.sidebar.number_input("Select max tokens", value=500)
minimum_new_tokens = st.sidebar.number_input("Select min tokens", value=0)
with st.sidebar:
    decoding_method = st.radio(
        "Select decoding method",
        ('greedy','sample')
    )
repetition_penalty = st.sidebar.number_input("Repetition penalty (Choose either 1 or 2)", min_value=1, max_value=2, value=1)
temperature = st.sidebar.number_input("Temperature (Choose a decimal number between 0 & 2)", min_value=0.0, max_value=2.0, step=0.3, value=0.5)
top_k = st.sidebar.number_input("Top K tokens (Choose an integer between 0 to 100)", min_value=0, max_value=100, step=10, value=50)
top_p = st.sidebar.number_input("Token probabilities (Choose a decimal number between 0 & 1)", min_value=0.0, max_value=1.0, step=0.1, value=0.5)

#@st.cache_data
def load_docs(files):
    st.info("`Reading doc ...`")
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Please provide txt or pdf file.', icon="⚠️")
    return all_text
         
    
@st.cache_resource
def create_retriever(_embeddings, splits):
    vectorstore = FAISS.from_texts(splits, _embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

@st.cache_resource
def split_texts(text, chunk_size, chunk_overlap, split_method):

    st.info("`Splitting doc ...`")

    split_method = "RecursiveCharacterTextSplitter"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to split document")
        st.stop()

    return splits


@st.cache_resource
def embed():
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large",model_kwargs={"device": "cpu"})
    return embeddings
    


def main():
    #global genai_api_key
    chat_history = []

# Use RecursiveCharacterTextSplitter as the default and only text splitter
    splitter_type = "RecursiveCharacterTextSplitter"
    embeddings = embed()
    #embeddings = HuggingFaceInstructEmbeddings()

    if 'genai_api_key' not in st.session_state:
        genai_api_key = st.text_input(
            'Please enter your GenAI API key', value="", placeholder="Enter the GenAI API key which begins with pak-")
        if genai_api_key:
            st.session_state.genai_api_key = genai_api_key
            os.environ["GENAI_API_KEY"] = genai_api_key
        else:
            return
    else:
        os.environ["GENAI_API_KEY"] = st.session_state.genai_api_key

    uploaded_files = st.file_uploader("Upload a PDF or TXT Document", type=[
                                      "pdf", "txt"], accept_multiple_files=True)

    if uploaded_files:
        # Check if last_uploaded_files is not in session_state or if uploaded_files are different from last_uploaded_files
        if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
            st.session_state.last_uploaded_files = uploaded_files
                 # Load and process the uploaded PDF or TXT files.
        loaded_text = load_docs(uploaded_files)
        total_text = len(loaded_text)
        st.write(f"Number of tokens: {total_text}")
        st.write("Documents uploaded and processed.")

        # Split the document into chunks
        splits = split_texts(loaded_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, split_method=splitter_type)

        # Display the number of text chunks
        num_chunks = len(splits)
        st.write(f"Number of text chunks: {num_chunks}")
        retriever = create_retriever(embeddings, splits)
        genai_api_key=st.session_state.genai_api_key
        creds = Credentials(api_key=genai_api_key, api_endpoint=genai_api_url)
        params = GenerateParams(decoding_method=decoding_method, temperature=temperature, max_new_tokens=maximum_new_tokens, min_new_tokens=minimum_new_tokens, repetition_penalty=repetition_penalty, top_k=top_k, top_p=top_p)
        llm=LangChainInterface(model="meta-llama/llama-2-70b-chat", params=params, credentials=creds)
        pre_prompt = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\nGenerate the next agent 
        response by answering the question. You are provided several documents with titles. If you cannot answer the question from the 
        given documents, please state I don't know.\n"""
        prompt = pre_prompt + "CONTEXT:\n\n{context}\n" +"Question : {question}" + "[\INST]"
        ll_prompt = PromptTemplate(template=prompt, input_variables=["context", "question"])
        chain = ConversationalRetrievalChain.from_llm(llm, retriever, combine_docs_chain_kwargs={"prompt": ll_prompt}, return_source_documents=False)
        st.write("Ready to answer questions.")
        
         # Question and answering
        user_question = st.text_input("Enter your question:")
        if user_question:
            with st.spinner("Working on it ..."):
                result = chain({"question": user_question, "chat_history": chat_history})
                st.write("Answer:", result['answer'])


if __name__ == "__main__":
    main()
    
