from src.helper import load_pdf, text_chunking, download_huggingface_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

extracted_data = load_pdf(data = 'data/')
text_chunks = text_chunking(extracted_data)
embeddings = download_huggingface_embeddings()

pc = Pinecone(api_key = PINECONE_API_KEY)

index_name = "carebot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric = 'cosine',
        spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
        )
    )

vs = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
    
)    