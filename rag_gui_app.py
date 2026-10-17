
import os
from langchain_postgres.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_openai.decisions import Choice, OpenAIDecisions
from sqlalchemy import create_engine
from dotenv import load_dotenv
import gradio as gr


load_dotenv()

# fail safe for wrong .env entry
def require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value

# Setup information statics
DB_USER = require("PG_USER")
DB_PASS = require("PG_PASSWORD")
DB_HOST = require("PG_HOST")
DB_PORT = require("PG_PORT")
DB_NAME = require("PG_DB")

EMBEDDING_MODEL = require("EMBEDDING_MODEL")
LLM_MODEL = require("LLM_MODEL")
DECISION_MODEL = require("DECISION_MODEL")
LLM_BASE_URL = require("LLM_BASE_URL")
LLM_API_KEY = require("LLM_API_KEY")

# Database Connection String
DATABASE_URL = f"postgresql+psycopg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

ollamaembeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL, # Same base URL as above
)

## multi RAGS
# One shared engine → all stores share the same connection pool
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)

## collection names
COLLECTIONS  = {"climate":"20230525_step-klima20-online","housing":"senstadt_step_broschuere_wohnen","economy":"senstadt_step-broschuere-wirtschaft"}

stores = {
    topic: PGVector(
        embeddings=ollamaembeddings,
        collection_name=name,
        connection=engine,          # shared, not a fresh string per instance
    )
    for topic, name in COLLECTIONS.items()
}

decisions = OpenAIDecisions(
    model=DECISION_MODEL,
    api_key=LLM_API_KEY,  
    base_url = LLM_BASE_URL, 
    )

def topic_decision(user_input):
    response = decisions.invoke(
        {
            "input": (
                '"' + user_input + '"'
            ),
            "questions":
            { 
                "topic": Choice(
                    instructions="Which topics belongs to the input?",
                    choices={
                        "housing": "über wohnung, housing. ",
                        "climate": "über Klima, Wetter.",
                        "economy": "über Wirtschaft, Geschäft, business",
                        "general": "was nicht gehört in Wohnung, Klima, Wirtschaft"
                        
                    },
                ),
                
            },
        }
    )
    
    return response.choices["topic"].choice

### for chat in GUI

def clean_content(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = [item.get('text', '') if isinstance(item, dict) else str(item)
                 for item in content]
        return " ".join(parts)
    return str(content)


def rag_answer(query, topic):
    """Retrieval + LLM. Topic is passed in, no longer decided here."""
    docs = None
    if topic in ("housing", "climate", "economy"):
        scored = stores[topic].similarity_search_with_score(query, k=4)
        top_score = scored[0][1] if scored else None
        print("topscore:",top_score)

        if top_score is not None and top_score <= 0.75:
            docs = [d for d, _ in scored]

    if docs:
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        prompt = "Answer in user's language with using the following context... Context: {context} Question: {question}"
        return llm.invoke(prompt.format(context=context, question=query)).content

    prompt = (
        "Answer in uses's language from your own general knowledge. "
        "State clearly that this answer is based on general knowledge, "
        "not on the user's documents.\n\nQuestion: {question}\nAnswer:"
    )
    return llm.invoke(prompt.format(question=query)).content

def classify(history, _st):
    """Fast-ish: decide the topic, show the category line."""
    query = None
    for m in reversed(history):
        if isinstance(m, dict) and m.get("role") == "user":
            query = clean_content(m.get("content"))
            break
    if not query:
        return history, None

    topic = topic_decision(query)
    cat = {"role": "assistant", "content": f"**User input category: {topic}**"}
    return history + [cat], {"topic": topic, "query": query}


def user(message, history, _st):
    """Instant: show prompt, clear input, reset state."""
    if not message:
        return message, history
    clean_msg = clean_content(message)
    return "", history + [{"role": "user", "content": clean_msg}], None

def bot(history, st):
    """Slow, SYNCHRONOUS: retrieval + LLM.
    Stays 'pending' the whole time, so the animation + elapsed time keep running."""
    if not st or not st.get("topic"):
        return history, st
    try:
        answer = rag_answer(st["query"], st["topic"])
    except Exception as e:
        import traceback
        answer = f"Error: {e}\n{traceback.format_exc()}"
    return history + [{"role": "assistant", "content": answer}], st

def main():
    # Initialize Gradio Interface
    with gr.Blocks(title="RAG Assistant") as demo:
        gr.Markdown("Routing RAG Document Chat")

        chatbot = gr.Chatbot(height=500)
        state = gr.State(None)

        with gr.Row():
            msg = gr.Textbox(placeholder="Ask a question about the document...", scale=4, container=False)
            btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            clear = gr.ClearButton([msg, chatbot])

        chain = [msg, chatbot, state], [msg, chatbot, state]

        msg.submit(user, *chain, queue=False).then(classify, [chatbot, state], [chatbot, state]).then(bot, [chatbot, state], [chatbot, state])

        btn.click(user, *chain, queue=False).then(classify, [chatbot, state], [chatbot, state]).then(bot, [chatbot, state], [chatbot, state])

    # Launch the app
    demo.launch(server_name="0.0.0.0", server_port=9090)

if __name__ == "__main__":
    main()
