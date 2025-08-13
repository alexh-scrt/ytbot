from ytbot.yt import (
    get_transcript,
    process_transcript,
)
from ytbot.llm import (
    initialize_ollama_llm,
    setup_embedding_model,
    create_qa_chain,
    get_params
)
from ytbot.store import (
    create_chroma_collection,
    chunk_transcript,
)
from ytbot.prompt import (
    create_qa_prompt_template,
)
from ytbot.qa import (
    generate_answer
)
from ytbot.llm import (
    Settings,
    LLMSettings,
    get_settings,
    get_params
)

processed_transcript = False

def answer_question(video_url, user_question):
    """
    Title: Answer User's Question

    Description:
    This function retrieves relevant context from the FAISS index based on the user’s query 
    and generates an answer using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
        user_question (str): The question posed by the user regarding the video.

    Returns:
        str: The answer to the user's question or a message indicating that the transcript 
             has not been fetched.
    """
    global fetched_transcript, processed_transcript

    settings = get_settings()
    # Check if the transcript needs to be fetched
    if not processed_transcript:
        if video_url:
            # Fetch and preprocess transcript
            fetched_transcript = get_transcript(video_url)
            processed_transcript = process_transcript(fetched_transcript)
        else:
            return "Please provide a valid YouTube URL."

    if processed_transcript and user_question:
        # Step 1: Chunk the transcript (only for Q&A)
        chunks = chunk_transcript(processed_transcript)

        # Step 2: Set up Ollama configuration
        params = get_params("qa", context_tokens=32768)  # if your model supports large ctx

        # Step 3: Initialize Ollama LLM for Q&A
        llm = initialize_ollama_llm(settings.llm, params)

        # Step 4: Create ChromaDB collection for transcript chunks (only needed for Q&A)
        embedding_model = setup_embedding_model(settings.llm)
        chroma_collection = create_chroma_collection(chunks, embedding_model)

        # Step 5: Set up the Q&A prompt and chain
        qa_prompt = create_qa_prompt_template()
        qa_chain = create_qa_chain(llm, qa_prompt)

        # Step 6: Generate the answer using ChromaDB collection
        answer = generate_answer(user_question, chroma_collection, embedding_model, qa_chain)
        return answer
    else:
        return "Please provide a valid question and ensure the transcript has been fetched."

