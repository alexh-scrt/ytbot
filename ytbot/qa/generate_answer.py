from ytbot.search.retriever import retrieve
from langchain.chains.llm import LLMChain

def generate_answer(question, chroma_collection, embedding_model, qa_chain: LLMChain, k=7):
    """
    Retrieve relevant context and generate an answer based on user input.

    Args:
        question: str
            The user's question.
        chroma_collection:
            The ChromaDB collection containing the embedded documents.
        embedding_model:
            The embedding model to use for the query.
        qa_chain: LLMChain
            The question-answering chain (LLMChain) to use for generating answers.
        k: int, optional (default=7)
            The number of relevant documents to retrieve.

    Returns:
        str: The generated answer to the user's question.
    """

    # Retrieve relevant context
    relevant_context = retrieve(question, chroma_collection, embedding_model, k=k)

    # Generate answer using the QA chain
    answer = qa_chain.predict(context=relevant_context, question=question)

    return answer
