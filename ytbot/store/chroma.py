
import chromadb  # For efficient vector storage and similarity search
from chromadb import Collection
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into manageable segments
from langchain_ollama import OllamaLLM, OllamaEmbeddings

def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    # Initialize the RecursiveCharacterTextSplitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split the transcript into chunks
    chunks = text_splitter.split_text(processed_transcript)
    return chunks



def create_chroma_collection(chunks, embedding_model: OllamaEmbeddings) -> Collection:
    """
    Create a ChromaDB collection from text chunks using the specified embedding model.
    
    :param chunks: List of text chunks
    :param embedding_model: The embedding model to use
    :return: ChromaDB collection
    """
    # Initialize ChromaDB client
    client = chromadb.Client()
    
    # Create or get a collection
    collection = client.get_or_create_collection(
        name="youtube_transcript",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Generate embeddings for all chunks
    embeddings = [embedding_model.embed_query(chunk) for chunk in chunks]
    
    # Add chunks and embeddings to the collection
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    
    return collection

