from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface.chat_models.huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
HF_TOKEN = os.environ.get("HF_TOKEN")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HF_TOKEN"] = HF_TOKEN

embeddings = download_huggingface_embeddings()

index_name = "carebot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
vs = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = HuggingFaceEndpoint(
        repo_id = "openai/gpt-oss-20b",
        huggingfacehub_api_token=HF_TOKEN,
        task='conversational',  
        max_new_tokens = 256,
        temperature = 0.3           
    )

chat = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(template = system_prompt, input_variables=['context','question'])

qa_chain=RetrievalQA.from_chain_type(
    llm = chat,
    chain_type = "stuff",
    retriever = retriever,
    chain_type_kwargs = {"prompt": prompt}
)




@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = qa_chain.invoke({"query": msg})
    print("Response : ", response["result"])
    return response["result"]



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)