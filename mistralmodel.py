from langchain_community.llms import HuggingFaceEndpoint
import os
from dotenv import load_dotenv
import requests
import os
import sys
from tempfile import TemporaryDirectory
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.globals import set_debug
from langchain.embeddings import CacheBackedEmbeddings
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_mistralai import MistralAIEmbeddings





# Load environment variables
load_dotenv('../resources/.env')

def get_api_key(key_name):
    """Get environment variables on the environment."""
    return os.environ.get(key_name)

# API Keys
huggingfacehub_api_token = get_api_key("HUGGINGFACEHUB_API_TOKEN")


# Set debug mode
set_debug(False)

# Initialize HuggingFace AI model
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.5, token=huggingfacehub_api_token
)

embedding = MistralAIEmbeddings(mistral_api_key=huggingfacehub_api_token)
embedding.model = "mistral-embed"

# Create a local file store for caching embeddings
store = LocalFileStore("./cache/")

# Setup embedder with cache
embedder = CacheBackedEmbeddings.from_bytes_store(embedding, store, namespace=embedding.model)

# Load and process documents for vector store creation
def load_and_process_document(file_path, chunk_size=2000, chunk_overlap=0):
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

# Define the document directory based on the environment
document_directory = "resource.txt"  # Update this path accordingly

if os.path.isfile(document_directory):
    document_texts = load_and_process_document(document_directory)
else:
    print(f"The document directory '{document_directory}' does not exist.")
    sys.exit(1)

# Create and populate the vector store
def create_vector_store(document_texts, embedder):
    if document_texts:
        if hasattr(embedder, 'embed_documents'):
            return FAISS.from_documents(document_texts, embedder)
        else:
            print("The embedder does not support document embedding.")
            return None
    else:
        print("No documents provided for vector store creation.")
        return None

print("Debug Point 1: Before creating vector store")
vectorstore = create_vector_store(document_texts, embedder)
print("Debug Point 2: After creating vector store")
retriever = vectorstore.as_retriever()
print("Debug Point 3: After creating retriever")
system_template = """
Sen, IT ve makine bakımı konularında net ve açıklayıcı bilgiler sunan bir yapay zeka asistanısın. Adın MahmutAI.
Yanıtların açıklayıcı ve net olsun. Yanıtlarında güler yüz ve emoji kullanarak kullanıcı deneyimini iyileştir.
Birazdan sorulacak soru hakkında bilgi sahibi değilsen, "bilmiyorum" diyerek dürüst bir yanıt ver.
Eğer bir kaynakça gösterebiliyorsan kaynakçayı belirt ve link göster ama asla bağlamı sunma.
Sondaki soruyu yanıtlamak için aşağıdaki bağlamı kullan: {context}." 
"""

system_message = SystemMessagePromptTemplate.from_template(system_template)
human_message = HumanMessagePromptTemplate.from_template("{question}")
messages = [system_message, human_message]
prompt = ChatPromptTemplate.from_messages(messages)


chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def main():
    while True:
        query = input("Sorgu: ")
        if query.lower() == "exit":
            break
        response = chain.invoke(query)
        print(f"Yanıt: {response}")

if __name__ == "__main__":
    main()