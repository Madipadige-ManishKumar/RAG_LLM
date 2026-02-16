from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from .data_preprocessing import EmbeddingManager,VectorStore,RAGRetriever,split_docs,GemmaRAGGenerator
import os
from dotenv import load_dotenv

load_dotenv()
apikey=os.getenv("apikey")





def setup_chatbot(bot_name, config, embedding_manager):
    print(f"\nInitializing {bot_name}...")

    pdf_folder = config["pdf_path"]

    if not os.path.exists(pdf_folder):
        print(f"PDF folder not found: {pdf_folder}")
        return None

    # Create / Load Vector Store
    vector_store = VectorStore(
        collection_name=f"{bot_name}_collection",
        persist_directory=config["db_path"]
    )

    # Ingest only if empty
    if vector_store.collection.count() == 0:
        print("Collection empty → ingesting PDFs...")

        loader = DirectoryLoader(
            pdf_folder,
            glob="*.pdf",
            loader_cls=PyPDFLoader
        )

        documents = loader.load()

        if not documents:
            print("No PDFs found.")
            return None

        # Split documents
        chunks = split_docs(documents)

        if not chunks:
            print("No chunks generated.")
            return None

        # Generate embeddings
        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)

        if len(chunks) != len(embeddings):
            raise ValueError("Mismatch between chunks and embeddings.")

        # Add to vector store (IMPORTANT: pass Document objects)
        vector_store.add_documents(
            document=chunks,
            embeddings=embeddings
        )

        print("Documents embedded and stored.")

    else:
        print("Existing collection loaded.")

    # Build retriever (NO score_threshold)
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedding_manager=embedding_manager
    )

    generator = GemmaRAGGenerator(apikey)

    return {
        "retriever": retriever,
        "generator": generator
    }

