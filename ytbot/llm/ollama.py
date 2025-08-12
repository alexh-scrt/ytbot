from langchain_ollama import OllamaLLM, OllamaEmbeddings  # For interacting with Ollama LLM and embeddings



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

