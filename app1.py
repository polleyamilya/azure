import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Prompt Engineer](https://youtube.com/@engineerprompt)')

load_dotenv()

def index_pdf(pdf):
    pdf_reader = PdfReader(pdf)

    indexed_data = []

    for page_num, page in enumerate(pdf_reader.pages, start=1):
        text = page.extract_text()
        indexed_data.append({"page": page_num, "text": text})

    return indexed_data

def search_answers(indexed_data, query):
    found_answers = []

    llm = OpenAI()
    chain = load_qa_chain(llm=llm, chain_type="stuff")

    for item in indexed_data:
        response = chain.run(input_documents=[{"text": item["text"]}], question=query)
        answer = response.get("answer", "")
        
        if answer:
            found_answers.append({"page": item["page"], "answer": answer})

    return found_answers

def main():
    st.header("Chat with PDF 💬")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Index PDF
        indexed_data = index_pdf(pdf)

        # Store indexed data
        store_name = pdf.name[:-4]
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(indexed_data, f)

        st.success(f"Successfully indexed PDF file!")

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            # Search for answers
            found_answers = search_answers(indexed_data, query)

            # Display answers
            for answer in found_answers:
                st.write(f"Page {answer['page']}:\n{answer['answer']}")

if __name__ == '__main__':
    main()
