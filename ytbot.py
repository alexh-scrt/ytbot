# Import necessary libraries for the YouTube bot
import gradio as gr
import re  #For extracting video id 
from youtube_transcript_api import YouTubeTranscriptApi  # For extracting transcripts from YouTube videos
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into manageable segments
from langchain_ollama import OllamaLLM, OllamaEmbeddings  # For interacting with Ollama LLM and embeddings
import chromadb  # For efficient vector storage and similarity search
from langchain.chains import LLMChain  # For creating chains of operations with LLMs
from langchain.prompts import PromptTemplate  # For defining prompt templates

def get_video_id(url):    
    # Regex pattern to match YouTube video URLs
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_transcript(url):
    # Extracts the video ID from the URL
    video_id = get_video_id(url)

    # Create a YouTubeTranscriptApi() object
    ytt_api = YouTubeTranscriptApi()
    
    # Fetch the list of available transcripts for the given YouTube video
    transcripts = ytt_api.list(video_id)
    
    transcript = ""
    for t in transcripts:
        # Check if the transcript's language is English
        if t.language_code == 'en':
            if t.is_generated:
                # If no transcript has been set yet, use the auto-generated one
                if len(transcript) == 0:
                    transcript = t.fetch()
            else:
                # If a manually created transcript is found, use it (overrides auto-generated)
                transcript = t.fetch()
                break  # Prioritize the manually created transcript, exit the loop
    
    return transcript if transcript else None


def process(transcript):
    # Initialize an empty string to hold the formatted transcript
    txt = ""
    
    # Loop through each entry in the transcript
    for i in transcript:
        try:
            # Append the text and its start time to the output string
            #txt += f"Text: {i['text']} Start: {i['start']}\n"
            txt += f"Text: {i.text} Start: {i.start}\n"
        except KeyError:
            # If there is an issue accessing 'text' or 'start', skip this entry
            pass
            
    # Return the processed transcript as a single string
    return txt

def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    # Initialize the RecursiveCharacterTextSplitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split the transcript into chunks
    chunks = text_splitter.split_text(processed_transcript)
    return chunks


def setup_ollama_config():
    # Define the model ID for the Ollama model being used
    model_id = "llama3.3:70b"
    
    # Define the base URL for Ollama service
    base_url = "http://localhost:11434"
    
    # Define the embedding model to use with Ollama
    embedding_model_id = "nomic-embed-text"
    
    # Return the configuration for later use
    return model_id, base_url, embedding_model_id

def define_parameters():
    # Return a dictionary containing the parameters for the Ollama model
    return {
        # Temperature for sampling (0 for deterministic/greedy)
        "temperature": 0.0,
        
        # Maximum number of tokens to generate
        "max_tokens": 900,
        
        # Top-p sampling parameter
        "top_p": 1.0,
    }


def initialize_ollama_llm(model_id, base_url, parameters):
    # Create and return an instance of the OllamaLLM with the specified configuration
    return OllamaLLM(
        model=model_id,                    # Set the model ID for the LLM
        base_url=base_url,                 # Set the base URL for Ollama service
        temperature=parameters.get("temperature", 0.0),  # Set temperature
        top_p=parameters.get("top_p", 1.0),             # Set top_p
        num_predict=parameters.get("max_tokens", 900)   # Set max tokens
    )



def setup_embedding_model(base_url, embedding_model_id):
    # Create and return an instance of OllamaEmbeddings with the specified configuration
    return OllamaEmbeddings(
        model=embedding_model_id,           # Set the embedding model ID
        base_url=base_url                  # Set the base URL for Ollama service
    )



def create_chroma_collection(chunks, embedding_model):
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


def create_summary_prompt():
    """
    Create a PromptTemplate for summarizing a YouTube video transcript.
    
    :return: PromptTemplate object
    """
    # Define the template for the summary prompt (Llama 3 format)
    template = """You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

Instructions:
1. Summarize the transcript in a single concise paragraph.
2. Ignore any timestamps in your summary.
3. Focus on the spoken content (Text) of the video.

Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.

Please summarize the following YouTube video transcript:

{transcript}

Summary:"""
    
    # Create the PromptTemplate object with the defined template
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template=template
    )
    
    return prompt


def create_summary_chain(llm, prompt, verbose=True):
    """
    Create an LLMChain for generating summaries.
    
    :param llm: Language model instance
    :param prompt: PromptTemplate instance
    :param verbose: Boolean to enable verbose output (default: True)
    :return: LLMChain instance
    """
    return LLMChain(llm=llm, prompt=prompt, verbose=verbose)


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

def create_qa_prompt_template():
    """
    Create a PromptTemplate for question answering based on video content.
    Returns:
        PromptTemplate: A PromptTemplate object configured for Q&A tasks.
    """
    
    # Define the template string (Llama 3 format)
    qa_template = """You are an expert assistant providing detailed and accurate answers based on the following video content. Your responses should be:
1. Precise and free from repetition
2. Consistent with the information provided in the video
3. Well-organized and easy to understand
4. Focused on addressing the user's question directly

If you encounter conflicting information in the video content, use your best judgment to provide the most likely correct answer based on context.

Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.

Relevant Video Context: {context}

Based on the above context, please answer the following question:
{question}

Answer:"""
    # Create the PromptTemplate object
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )
    return prompt_template


def create_qa_chain(llm, prompt_template, verbose=True):
    """
    Create an LLMChain for question answering.

    Args:
        llm: Language model instance
            The language model to use in the chain (e.g., WatsonxGranite).
        prompt_template: PromptTemplate
            The prompt template to use for structuring inputs to the language model.
        verbose: bool, optional (default=True)
            Whether to enable verbose output for the chain.

    Returns:
        LLMChain: An instantiated LLMChain ready for question answering.
    """
    
    return LLMChain(llm=llm, prompt=prompt_template, verbose=verbose)


def generate_answer(question, chroma_collection, embedding_model, qa_chain, k=7):
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


# Initialize an empty string to store the processed transcript after fetching and preprocessing
processed_transcript = ""

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
        processed_transcript = process(fetched_transcript)
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

    # Check if the transcript needs to be fetched
    if not processed_transcript:
        if video_url:
            # Fetch and preprocess transcript
            fetched_transcript = get_transcript(video_url)
            processed_transcript = process(fetched_transcript)
        else:
            return "Please provide a valid YouTube URL."

    if processed_transcript and user_question:
        # Step 1: Chunk the transcript (only for Q&A)
        chunks = chunk_transcript(processed_transcript)

        # Step 2: Set up Ollama configuration
        model_id, base_url, embedding_model_id = setup_ollama_config()

        # Step 3: Initialize Ollama LLM for Q&A
        llm = initialize_ollama_llm(model_id, base_url, define_parameters())

        # Step 4: Create ChromaDB collection for transcript chunks (only needed for Q&A)
        embedding_model = setup_embedding_model(base_url, embedding_model_id)
        chroma_collection = create_chroma_collection(chunks, embedding_model)

        # Step 5: Set up the Q&A prompt and chain
        qa_prompt = create_qa_prompt_template()
        qa_chain = create_qa_chain(llm, qa_prompt)

        # Step 6: Generate the answer using ChromaDB collection
        answer = generate_answer(user_question, chroma_collection, embedding_model, qa_chain)
        return answer
    else:
        return "Please provide a valid question and ensure the transcript has been fetched."



with gr.Blocks() as interface:

    gr.Markdown(
        "<h2 style='text-align: center;'>YouTube Video Summarizer and Q&A</h2>"
    )

    # Input field for YouTube URL
    video_url = gr.Textbox(label="YouTube Video URL", placeholder="Enter the YouTube Video URL")
    
    # Outputs for summary and answer
    summary_output = gr.Textbox(label="Video Summary", lines=5)
    question_input = gr.Textbox(label="Ask a Question About the Video", placeholder="Ask your question")
    answer_output = gr.Textbox(label="Answer to Your Question", lines=5)

    # Buttons for selecting functionalities after fetching transcript
    summarize_btn = gr.Button("Summarize Video")
    question_btn = gr.Button("Ask a Question")

    # Display status message for transcript fetch
    transcript_status = gr.Textbox(label="Transcript Status", interactive=False)

    # Set up button actions
    summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
    question_btn.click(answer_question, inputs=[video_url, question_input], outputs=answer_output)

# Launch the app with specified server name and port
interface.launch(server_name="0.0.0.0", server_port=7860)