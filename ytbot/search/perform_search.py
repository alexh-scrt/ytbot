
def perform_similarity_search(chroma_collection, embedding_model, query, k=3):
    """
    Search for specific queries within the embedded transcript using the ChromaDB collection.
    
    :param chroma_collection: The ChromaDB collection containing embedded text chunks
    :param embedding_model: The embedding model to use for the query
    :param query: The text input for the similarity search
    :param k: The number of similar results to return (default is 3)
    :return: List of similar results
    """
    # Generate embedding for the query
    query_embedding = embedding_model.embed_query(query)
    
    # Perform the similarity search using ChromaDB
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    # Extract and return the documents
    return results['documents'][0] if results['documents'] else []


