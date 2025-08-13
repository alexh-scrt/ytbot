# Import necessary libraries for the YouTube bot
import gradio as gr
from pathlib import Path
from ytbot.llm import (
    Settings
)
from ytbot import (
    answer_question,
    summarize_video
)

def load_css():
    """Load the YouTube-inspired CSS styling"""
    css_path = Path(__file__).parent.parent / "static" / "youtube_style.css"
    
    # Fallback inline CSS if file doesn't exist
    if css_path.exists():
        return css_path.read_text()
    else:
        # Inline fallback CSS
        return """
        :root {
            --yt-spec-base-background: #0f0f0f;
            --yt-spec-raised-background: #212121;
            --yt-spec-text-primary: #f1f1f1;
            --yt-spec-text-secondary: #aaaaaa;
            --yt-spec-outline: #303030;
            --yt-spec-call-to-action: #cc0000;
            --yt-spec-brand-button-text: #065fd4;
        }
        
        .gradio-container {
            background-color: var(--yt-spec-base-background) !important;
            color: var(--yt-spec-text-primary) !important;
            font-family: "YouTube Sans", "Roboto", sans-serif !important;
        }
        
        .gradio-container .gr-textbox {
            background-color: var(--yt-spec-raised-background) !important;
            border: 1px solid var(--yt-spec-outline) !important;
            border-radius: 20px !important;
            color: var(--yt-spec-text-primary) !important;
        }
        
        .gradio-container .gr-button.primary {
            background-color: var(--yt-spec-call-to-action) !important;
            border-color: var(--yt-spec-call-to-action) !important;
            color: white !important;
            border-radius: 18px !important;
        }
        
        .gradio-container .gr-column {
            background-color: var(--yt-spec-raised-background) !important;
            border-radius: 12px !important;
            padding: 20px !important;
            border: 1px solid var(--yt-spec-outline) !important;
        }
        """

def launch_ui(settings: Settings):
    # Load the custom CSS
    custom_css = load_css()
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as interface:

        gr.Markdown(
            "<h2 style='text-align: center;'>🎥 YouTube Video Summarizer and Q&A</h2>"
        )

        # Input field for YouTube URL (full width at top)
        video_url = gr.Textbox(
            label="YouTube Video URL", 
            placeholder="Paste your YouTube video link here...",
            container=True,
            elem_classes=["url-input"]
        )
        
        # Two column layout with equal heights
        with gr.Row(equal_height=True):
            # Left column - Summary
            with gr.Column(scale=1, elem_classes=["summary-column"]):
                gr.Markdown("### 📝 Video Summary")
                summary_output = gr.Textbox(
                    label="Summary", 
                    lines=10,
                    placeholder="Video summary will appear here...",
                    interactive=False,
                    elem_classes=["summary-output"]
                )
                summarize_btn = gr.Button(
                    "Summarize Video", 
                    variant="primary", 
                    size="lg",
                    elem_classes=["summarize-btn"]
                )
            
            # Right column - Q&A
            with gr.Column(scale=1, elem_classes=["qa-column"]):
                gr.Markdown("### 💬 Ask Questions")
                question_input = gr.Textbox(
                    label="Question", 
                    placeholder="Ask your question about the video...",
                    lines=2,
                    elem_classes=["question-input"]
                )
                answer_output = gr.Textbox(
                    label="Answer", 
                    lines=8,
                    placeholder="Answer will appear here...",
                    interactive=False,
                    elem_classes=["answer-output"]
                )
                question_btn = gr.Button(
                    "Ask Question", 
                    variant="primary", 
                    size="lg",
                    elem_classes=["question-btn"]
                )

        # Status message at the bottom (full width)
        transcript_status = gr.Textbox(
            label="Status", 
            interactive=False,
            visible=False,  # Hidden by default, can be shown for debugging
            elem_classes=["status-output"]
        )

        # Set up button actions
        summarize_btn.click(
            summarize_video,
            inputs=video_url, 
            outputs=summary_output
        )
        
        question_btn.click(
            answer_question, 
            inputs=[video_url, question_input], 
            outputs=answer_output
        )

    # Launch the app with specified server name and port
    interface.launch(server_name="0.0.0.0", server_port=7860)