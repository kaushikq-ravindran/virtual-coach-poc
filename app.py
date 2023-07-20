import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from PIL import Image
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine, DeanonymizeEngine
# from presidio_anonymizer.entities import OperatorResult, OperatorConfig


# Imports to use open source models
# from langchain.llms import HuggingFaceHub
# from langchain.embeddings import HuggingFaceInstructEmbeddings

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    img = Image.open('images/deloitte-title.png')
    st.set_page_config(page_title="Deloitte Virtual Coach",
                       page_icon=img)
    st.write(css, unsafe_allow_html=True)

    # Create configuration containing engine name and models for Anonymizing data
    configuration = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    }

    # Create NLP engine based on configuration
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()
    # the languages are needed to load country-specific recognizers 
    # to find and mask private data such as phones, passport numbers, etc.
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine,
                          supported_languages=["en"]) 
    anonymizer = AnonymizerEngine()
    deanonymizer = DeanonymizeEngine()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

        st.header("Deloitte Virtual Coach")

    user_question = st.text_input("Upload your Documents and say Hi to your Private Coach!")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your work-related documents")
        pdf_docs = st.file_uploader(
            "You can upload any document within the Deloitte eco-system that you need help with. \n \n For example - Expectation Frameworks, Impact Statements etc.", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # results = analyzer.analyze(text=raw_text, language='en')

                # for res in results:
                #     print(res)

                #     anonymized_text = anonymizer.anonymize(text=raw_text, analyzer_results=results).text
                #     print(anonymized_text)

                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
                # Code to retreive original data - still in works
                # deAnonymizedText = deanonymizer.deanonymize(anonymized_text, entities=[
                #                                     OperatorResult(start=11, end=55, entity_type="PERSON"),
                #                                      ],
                #                                     operators={"DEFAULT": OperatorConfig("decrypt", {"key": "WmZq4t7w!z%C&F)J"})})                
                # print(deAnonymizedText)

if __name__ == '__main__':
    main()
