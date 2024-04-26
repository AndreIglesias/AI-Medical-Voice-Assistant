from . import ollama_model, console, OPENAI_API_KEY, template, context_template
from operator import itemgetter
import chromadb
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
# Langchain imports
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
# Ollama imports
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
# OpenAI imports
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
# Loaders
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

# ==================================================================================================

# Definition of the model and embeddings
if ollama_model.startswith("gpt"):
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=ollama_model)
    embeddings = OpenAIEmbeddings()
else:
    model = Ollama(model=ollama_model)
    embeddings = OllamaEmbeddings(model=ollama_model)

# Definition of the output parser
parser = StrOutputParser()

# ==================================================================================================

# Vector store and retriever settings

vector_store_directory = "chroma_store"
collection_name = "pdf-rag"

chroma_settings = chromadb.Settings(persist_directory=vector_store_directory)

chroma_client = chromadb.Client(chroma_settings)

vectorstore = Chroma(
    persist_directory=vector_store_directory,
    collection_name=collection_name,
    embedding_function=embeddings,
    client=chroma_client,
)
# ==================================================================================================

def learn_pdf(pdf_path, question="What is the main idea of this document?"):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()  # Load all pages of the PDF

    text_splitter = CharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    document_chunks = text_splitter.split_documents(documents)

    vectorstore.add_documents(document_chunks)
    vectorstore.persist()

    temp_vectorstore = Chroma(
        persist_directory=vector_store_directory,
        collection_name=f"pdf-temp-{hash(pdf_path)}",
        embedding_function=embeddings,
        client=chroma_client,
    )

    temp_vectorstore.add_documents(document_chunks)
    temp_vectorstore.persist()

    retriever = temp_vectorstore.as_retriever()

    chat_prompt = ChatPromptTemplate.from_template(context_template)

    # Define a chain that fetches relevant context and produces a response
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | chat_prompt
        | model
        | parser
    )

    response = chain.invoke(question)
    return response


def process_urls(urls, question):

    # Split the input URLs into a list
    url_list = urls.split("\n")
    
    # Load documents from the URLs using a WebBaseLoader
    docs = []
    for url in url_list:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())  # Load all documents from the URL
        print(docs)

    # Split the documents into chunks for vectorization
    text_splitter = CharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    document_chunks = text_splitter.split_documents(docs)
    
    # Create a Chroma vector store from the document chunks
    vectorstore = Chroma.from_documents(
        documents=document_chunks,
        embedding=embeddings,
        collection_name="web-rag"
    )

    # Create a retriever from the Chroma vector store
    retriever = vectorstore.as_retriever()

    # Create a ChatPromptTemplate for the context
    chat_prompt = ChatPromptTemplate.from_template(context_template)

    # Define a chain that fetches relevant context and produces a response
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | chat_prompt
        | model
        | parser
    )

    # Get the response to the question
    response = chain.invoke(question)

    return response

# Load PDF and split into chunks
def process_pdf(pdf_path, question):
    # Load and split the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()  # Load all pages of the PDF
    print(documents)

    # Split the documents into smaller chunks for better vectorization
    text_splitter = CharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    document_chunks = text_splitter.split_documents(documents)

    # Create a Chroma vector store from the document chunks
    vectorstore = Chroma.from_documents(
        documents=document_chunks,
        embedding=embeddings,
        collection_name="pdf-rag"
    )

    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever()

    # Define a new chain to process the question and retrieve relevant chunks
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | PromptTemplate(input_variables=["context", "question"], template=context_template)
        | model  # Model being used for text completion/chat
        | parser  # Parse the response into a more readable format
    )

    # Get the response from the chain
    response = chain.invoke({"question": question})

    # Print or return the response as needed
    print(f"Answer: {response}")

    return response


def get_response(input_text):
    # Definition of the prompt
    prompt = PromptTemplate(input_variables=["history", "input"], template=template)
    
    # Definition of the conversation chain
    chain = ConversationChain(
        prompt=prompt,
        verbose=False,
        memory=ConversationBufferMemory(ai_prefix="Assistant IA: ", user_prefix="Utilisateur: "),
        llm=Ollama(model=ollama_model),
    )

    console.print(f"[cyan]User: {input_text}")
    response = chain.predict(input=input_text)
    console.print(f"[cyan]Assistant IA: {response}")
    if response.startswith("Assistant IA: "):
        response = response[len("Assistant IA: "):].strip()
    return response

