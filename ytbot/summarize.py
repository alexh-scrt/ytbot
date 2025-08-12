from ytbot.yt.process import (
    get_transcript,
    process_transcript
)
from ytbot.llm import (
    setup_ollama_config,
    initialize_ollama_llm,
    define_parameters,
    setup_embedding_model,
    create_summary_chain,
    create_qa_chain
)
from ytbot.prompt import (
    create_qa_prompt_template,
    create_summary_prompt
)

def summarize_video(video_url):
    """
    Title: Summarize Video

    Description:
    This function generates a summary of the video using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.

    Returns:
        str: The generated summary of the video or a message indicating that no transcript is available.
    """
    global fetched_transcript, processed_transcript
    
    
    if video_url:
        # Fetch and preprocess transcript
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process_transcript(fetched_transcript)
    else:
        return "Please provide a valid YouTube URL."

    if processed_transcript:
        # Step 1: Set up Ollama configuration
        model_id, base_url, embedding_model_id = setup_ollama_config()

        # Step 2: Initialize Ollama LLM for summarization
        llm = initialize_ollama_llm(model_id, base_url, define_parameters())

        # Step 3: Create the summary prompt and chain
        summary_prompt = create_summary_prompt()
        summary_chain = create_summary_chain(llm, summary_prompt)

        # Step 4: Generate the video summary
        summary = summary_chain.run({"transcript": processed_transcript})
        return summary
    else:
        return "No transcript available. Please fetch the transcript first."

