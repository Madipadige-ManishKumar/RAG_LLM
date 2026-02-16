from .data_preprocessing import EmbeddingManager,VectorStore,RAGRetriever,process_all_pdfs,split_docs,GemmaRAGGenerator
from .chatbot_setup import setup_chatbot
import os
from dotenv import load_dotenv



load_dotenv()
apikey=os.getenv("apikey")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(BASE_DIR, "pdfs")

chatbot_configs = {
    "Computer-Network-bot": {
        "pdf_path": os.path.join(base_path, "set1"),
        "db_path": os.path.join(BASE_DIR, "./db/CN"),
        "collection_name":"CN",
    },
    "ML-bot": {
        "pdf_path": os.path.join(base_path, "set2"),
        "db_path":os.path.join(BASE_DIR, "./db/ML"),
        "collection_name":"ML",
    },
    "Os_bot": {
        "pdf_path": os.path.join(base_path, "set3"),
        "db_path": os.path.join(BASE_DIR, "./db/OS"),
        "collection_name":"OS",
    },
    "DBMS_bot": {
        "pdf_path": os.path.join(base_path, "set4"),
        "db_path": os.path.join(BASE_DIR, "./db/DBMS"),
        "collection_name":"DBMS",
    },
}

chatbots = {}


embedding_manager = EmbeddingManager()
generator = GemmaRAGGenerator(apikey) 


def initialize_chatbots(botname):
    print("Initializing chatbot...")

    vector_store = VectorStore(f"{botname}_collection",chatbot_configs[botname]["db_path"])
    

    retriever = RAGRetriever(vector_store, embedding_manager)

    chatbots[botname] = {
        "retriever": retriever,
        "generator": generator
    }

    print("All chatbots loaded.\n")
def build_chatbots():
    for bot_name, config in chatbot_configs.items():
        bot_instance = setup_chatbot(bot_name, config,embedding_manager=embedding_manager)
        if bot_instance:
            chatbots[bot_name] = bot_instance




def query_chatbot(bot_name, question):
    print(chatbot_configs[bot_name]["collection_name"])
    initialize_chatbots(bot_name)
    bot = chatbots.get(bot_name)

    if not bot:
        return "Chatbot not initialized."

    retriever = bot["retriever"]
    generator = bot["generator"]

    retrieved_docs = retriever.retrieve(question)

    if not retrieved_docs:
        return "No relevant context found."

    answer = generator.generate_answer(
        question,
        retrieved_docs
    )

    return answer
