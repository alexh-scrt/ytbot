from ytbot.yt.process import (
    get_transcript,
    process_transcript
)
from ytbot.llm import (
    initialize_ollama_llm,
    create_summary_chain,
    get_settings,
)
from ytbot.prompt import (
    create_summary_prompt
)
from ytbot.llm import (
    Settings,
    get_params
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
    
    settings = get_settings()
    if video_url:
        # Fetch and preprocess transcript
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process_transcript(fetched_transcript)
    else:
        return "Please provide a valid YouTube URL."

    if processed_transcript:
        # Step 1: Set up Ollama configuration
        params = get_params("summarization", context_tokens=16384, overrides={"num_predict": 384})

        # Step 2: Initialize Ollama LLM for summarization
        llm = initialize_ollama_llm(settings.llm, params)

        # Step 3: Create the summary prompt and chain
        summary_prompt = create_summary_prompt()
        summary_chain = create_summary_chain(llm, summary_prompt)

        # Step 4: Generate the video summary
        summary = summary_chain.run({"transcript": processed_transcript})
        return summary
    else:
        return "No transcript available. Please fetch the transcript first."

