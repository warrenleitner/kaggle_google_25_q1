# The Ghostwriter

A multi-agent system that learns your specific writing style and transforms audio notes into polished prose that sounds exactly like you.

## Setup

1.  **Install Dependencies**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install google-genai
    ```

2.  **Set API Key**:
    ```bash
    export GOOGLE_API_KEY="your_api_key_here"
    ```

3.  **Add Style Examples**:
    Place 5-10 text files (`.txt`) of your past writing in `data/style_examples/`.
    (A sample `example_style.txt` is provided).

## Usage

### Phase 1: Style Architecture (Setup)
Run this once to analyze your writing style and generate the system prompt.
```bash
python style_architect.py
```
This will create `data/style_prompt.txt` and `data/style_snippets.json`.

### Phase 2: The Ghostwriter (Run)
Run the main application to process audio notes.
```bash
python ghostwriter_app.py
```
You will be prompted to enter the path to an audio file (or type 'test' for a dummy run).

## Architecture

*   **Listener Agent**: Transcribes audio using Gemini Multimodal.
*   **Editor Agent**: Cleans up the transcript (removes fillers, fixes grammar).
*   **Ghostwriter Agent**: Rewrites the draft using your Style System Prompt and RAG (retrieved snippets).
*   **Critic Agent**: Evaluates the output and provides feedback in a loop until the quality meets the threshold.
