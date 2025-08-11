# YouTube Video Summarizer & Q&A Bot

A powerful AI-powered tool that extracts transcripts from YouTube videos, generates concise summaries, and answers questions about the video content using local LLMs.

## 🚀 Features

- **Automatic Transcript Extraction**: Fetches English transcripts from YouTube videos (both manual and auto-generated)
- **Video Summarization**: Generates concise, informative summaries of video content
- **Interactive Q&A**: Ask questions about the video content and get accurate, context-aware answers
- **Semantic Search**: Uses ChromaDB for efficient vector storage and similarity search
- **Local LLM Integration**: Powered by Ollama with Llama 3.3 70B model for privacy and control
- **Web Interface**: Clean, user-friendly Gradio interface accessible from any browser

## 📋 Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running locally
- Llama 3.3 70B model pulled in Ollama
- Nomic Embed Text model for embeddings

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ytbot.git
cd ytbot
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up Ollama models**

Make sure Ollama is installed and running, then pull the required models:
```bash
ollama pull llama3.3:70b
ollama pull nomic-embed-text
```

## 🎯 Usage

1. **Start the application**
```bash
python ytbot.py
```

2. **Access the web interface**

Open your browser and navigate to:
```
http://localhost:7860
```

3. **Use the application**
   - Paste a YouTube video URL in the input field
   - Click "Summarize Video" to get a concise summary
   - Enter a question and click "Ask a Question" to get answers about the video content

## 🏗️ Architecture

### Core Components

- **Transcript Extraction**: Uses `youtube-transcript-api` to fetch video transcripts
- **Text Processing**: Implements chunking with `RecursiveCharacterTextSplitter` for optimal context management
- **Vector Database**: ChromaDB for efficient semantic search and retrieval
- **LLM Integration**: Ollama for local language model inference
- **Web Framework**: Gradio for the interactive user interface

### Key Functions

- `get_transcript()`: Extracts transcript from YouTube video URL
- `process()`: Formats transcript with timestamps
- `chunk_transcript()`: Splits text into manageable chunks
- `summarize_video()`: Generates video summary using LLM
- `answer_question()`: Retrieves relevant context and answers user queries

## ⚙️ Configuration

The application uses the following default settings:

- **LLM Model**: Llama 3.3 70B
- **Embedding Model**: nomic-embed-text
- **Ollama Base URL**: http://localhost:11434
- **Temperature**: 0.0 (deterministic responses)
- **Max Tokens**: 900
- **Chunk Size**: 200 characters
- **Chunk Overlap**: 20 characters

These settings can be modified in the respective configuration functions within `ytbot.py`.

## 🌐 Network Configuration

The application runs on:
- **Host**: 0.0.0.0 (accessible from all network interfaces)
- **Port**: 7860

To change these settings, modify the last line in `ytbot.py`:
```python
interface.launch(server_name="0.0.0.0", server_port=7860)
```

## 📦 Dependencies

Key dependencies include:
- `gradio`: Web interface framework
- `youtube-transcript-api`: YouTube transcript extraction
- `langchain`: LLM orchestration framework
- `langchain-ollama`: Ollama integration for LangChain
- `chromadb`: Vector database for semantic search
- `ollama`: Python client for Ollama

See `requirements.txt` for the complete list of dependencies.

## 🔧 Troubleshooting

### Common Issues

1. **Ollama connection error**
   - Ensure Ollama is running: `ollama serve`
   - Check if the models are downloaded: `ollama list`

2. **No transcript available**
   - Some videos may not have transcripts enabled
   - The tool prioritizes manually created transcripts over auto-generated ones

3. **Memory issues with large videos**
   - Adjust chunk size in `chunk_transcript()` function
   - Consider using a smaller model if running on limited hardware

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Powered by [Ollama](https://ollama.ai/)
- UI created with [Gradio](https://gradio.app/)
- Vector search by [ChromaDB](https://www.trychroma.com/)