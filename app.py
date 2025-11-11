from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

embeddings = download_huggingface_embeddings()

index_name = "carebot" 

vs = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k":3})

temperature = 0.5


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",       
    temperature=temperature,
    max_output_tokens=512,
    google_api_key=GOOGLE_API_KEY  
)

memory = ConversationBufferMemory(
    memory_key="chat_history", 
    input_key="question",
    output_key="answer",
    return_messages=True
)


prompt = PromptTemplate(
    template=system_prompt,
    input_variables=["chat_history", "context", "question"]
)

conv_rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=False,
    get_chat_history=lambda h: h
)



@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User:", msg)
    result = conv_rag_chain.invoke({"question": msg})
    answer = result["answer"]
    print("Response:", answer)
    return answer

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)