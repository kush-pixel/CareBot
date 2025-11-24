from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from src.helper import download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from typing import List
from pydantic import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from src.prompt import *
import os

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")
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

class CrossEncoderRerankRetriever(BaseRetriever):
    base_retriever: BaseRetriever = Field(...)     
    model_name: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    top_k: int = Field(default=4)

    
    cross_encoder: CrossEncoder = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        
        self.cross_encoder = CrossEncoder(self.model_name)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
      
        docs = self.base_retriever.get_relevant_documents(query)
        if not docs:
            return []

       
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.cross_encoder.predict(pairs)

        
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in ranked[: self.top_k]]
        return top_docs

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)

# retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k":3})

base_retriever = vs.as_retriever(
    search_kwargs={"k": 10}  
)


retriever = CrossEncoderRerankRetriever(
    base_retriever=base_retriever,
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k=4  
)

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
    # memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=False,
    get_chat_history=lambda h: h
)



@app.route("/")
def index():
    history = session.get("chat_history", [])
    return render_template('chat.html', chat_history = history)

# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     print("User:", msg)

#     # Get previous chat history from session (string list)
#     history = session.get("chat_history", [])

#     # Format history as a simple string (you can customize format later)
#     # e.g., "User: ...\nBot: ...\n..."
#     history_str = ""
#     for turn in history:
#         role, text = turn
#         if role == "user":
#             history_str += f"User: {text}\n"
#         else:
#             history_str += f"Bot: {text}\n"

#     # Call the chain with explicit chat_history
#     result = conv_rag_chain.invoke({
#         "question": msg,
#         "chat_history": history_str
#     })

#     answer = result["answer"]
#     print("Response:", answer)

#     # Append this turn to the session history
#     history.append(("user", msg))
#     history.append(("bot", answer))
#     session["chat_history"] = history

#     return answer

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User:", msg)

    # Get previous chat history from session
    history = session.get("chat_history", [])

    # Format for the prompt
    history_str = ""
    for role, text in history:
        if role == "user":
            history_str += f"User: {text}\n"
        else:
            history_str += f"CareBot: {text}\n"

    # Call the chain
    result = conv_rag_chain.invoke({
        "question": msg,
        "chat_history": history_str
    })

    answer = result["answer"]
    print("Response:", answer)

    # Update session chat history
    history.append(("user", msg))
    history.append(("bot", answer))
    session["chat_history"] = history

    return answer



@app.route("/clear", methods=["POST"])
def clear_chat():
    # Remove chat history from session
    session.pop("chat_history", None)
    session.pop("current_chat_index", None)
    return jsonify({"status": "ok"})


# @app.route("/save_chat", methods=["POST"])
# def save_chat():
#     # Name from frontend (popup)
#     data = request.get_json(silent=True) or {}
#     name = (data.get("name") or "").strip()

#     history = session.get("chat_history", [])
#     if not history:
#         return jsonify({"status": "empty", "message": "No chat history to save."})

#     # Format as plain text transcript
#     lines = []
#     for role, text in history:
#         role_label = "User" if role == "user" else "CareBot"
#         lines.append(f"{role_label}: {text}")
#     content = "\n".join(lines)

#     saved_chats = session.get("saved_chats", [])

#     if not name:
#         name = f"Chat {len(saved_chats) + 1}"

#     saved_chats.append({
#         "name": name,
#         "content": content
#     })
#     session["saved_chats"] = saved_chats

#     # 🔹 After saving, clear current chat history so next chat is fresh
#     session.pop("chat_history", None)

#     return jsonify({"status": "ok", "message": "Chat saved successfully.", "name": name})

@app.route("/save_chat", methods=["POST"])
def save_chat():
    # Name from frontend (popup)
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()

    history = session.get("chat_history", [])
    if not history:
        return jsonify({"status": "empty", "message": "No chat history to save."})

    # Format as plain text transcript
    lines = []
    for role, text in history:
        role_label = "User" if role == "user" else "CareBot"
        lines.append(f"{role_label}: {text}")
    content = "\n".join(lines)

    saved_chats = session.get("saved_chats", [])
    current_index = session.get("current_chat_index", None)

    # 🔹 If we are editing an existing saved chat, update it
    if isinstance(current_index, int) and 0 <= current_index < len(saved_chats):
        # Update name only if user provided a new one
        if name:
            saved_chats[current_index]["name"] = name
        # Always update content to latest conversation
        saved_chats[current_index]["content"] = content
        updated_name = saved_chats[current_index]["name"]
    else:
        # 🔹 New chat: append
        if not name:
            name = f"Chat {len(saved_chats) + 1}"
        saved_chats.append({
            "name": name,
            "content": content
        })
        current_index = len(saved_chats) - 1
        updated_name = name

    session["saved_chats"] = saved_chats

    # After saving, clear current chat + clear current_chat_index
    session.pop("chat_history", None)
    session.pop("current_chat_index", None)

    return jsonify({
        "status": "ok",
        "message": "Chat saved successfully.",
        "name": updated_name,
        "index": current_index
    })


# @app.route("/load_chat/<int:chat_index>", methods=["GET"])
# def load_chat(chat_index):
#     saved_chats = session.get("saved_chats", [])
#     if 0 <= chat_index < len(saved_chats):
#         chat = saved_chats[chat_index]
#         content = chat.get("content", "")

#         # Reconstruct chat_history from the saved text
#         history = []
#         for line in content.splitlines():
#             line = line.strip()
#             if line.startswith("User: "):
#                 text = line[len("User: "):]
#                 history.append(("user", text))
#             elif line.startswith("CareBot: "):
#                 text = line[len("CareBot: "):]
#                 history.append(("bot", text))

#         session["chat_history"] = history

#     # Redirect back to main chat, where history will be rendered
#     return redirect(url_for("index"))

@app.route("/load_chat/<int:chat_index>", methods=["GET"])
def load_chat(chat_index):
    saved_chats = session.get("saved_chats", [])
    if 0 <= chat_index < len(saved_chats):
        chat = saved_chats[chat_index]
        content = chat.get("content", "")

        # Reconstruct chat_history from the saved text
        history = []
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("User: "):
                text = line[len("User: "):]
                history.append(("user", text))
            elif line.startswith("CareBot: "):
                text = line[len("CareBot: "):]
                history.append(("bot", text))

        session["chat_history"] = history
        # 🔹 Remember which saved chat is currently being edited
        session["current_chat_index"] = chat_index

    return redirect(url_for("index"))


# @app.route("/delete_chat/<int:chat_index>", methods=["POST"])
# def delete_chat(chat_index):
#     saved_chats = session.get("saved_chats", [])

#     if 0 <= chat_index < len(saved_chats):
#         # Remove the selected chat
#         del saved_chats[chat_index]
#         session["saved_chats"] = saved_chats
#         return jsonify({"status": "ok"})

#     return jsonify({"status": "error", "message": "Invalid chat index."}), 400

@app.route("/delete_chat/<int:chat_index>", methods=["POST"])
def delete_chat(chat_index):
    saved_chats = session.get("saved_chats", [])

    if 0 <= chat_index < len(saved_chats):
        del saved_chats[chat_index]
        session["saved_chats"] = saved_chats
        # If we were editing some chat, reset that state
        session.pop("current_chat_index", None)
        return jsonify({"status": "ok"})

    return jsonify({"status": "error", "message": "Invalid chat index."}), 400


@app.route("/saved_chats", methods=["GET"])
def saved_chats():
    chats = session.get("saved_chats", [])
    return render_template("saved_chats.html", saved_chats=chats)

@app.route("/clear_saved_chats", methods=["POST"])
def clear_saved_chats():
    # Remove all saved chats
    session.pop("saved_chats", None)
    # Also clear current chat history memory
    session.pop("chat_history", None)
    session.pop("current_chat_index", None)
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)