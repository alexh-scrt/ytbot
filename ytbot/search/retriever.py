
def retrieve(query, chroma_collection, embedding_model, k=7):
    """
    Retrieve relevant context from the ChromaDB collection based on the user's query.

    Parameters:
        query (str): The user's query string.
        chroma_collection: The ChromaDB collection containing the embedded documents.
        embedding_model: The embedding model to use for the query.
        k (int, optional): The number of most relevant documents to retrieve (default is 7).

    Returns:
        list: A list of the k most relevant documents (or document chunks).
    """
    # Generate embedding for the query
    query_embedding = embedding_model.embed_query(query)
    
    # Query the collection
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    # Return the documents
    return results['documents'][0] if results['documents'] else []


