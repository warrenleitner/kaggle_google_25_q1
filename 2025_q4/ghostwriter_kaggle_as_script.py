# %% [markdown]
# Ghostwriter: A Style-Mimicking, stream of conciousness friendly writing assistant using Google Gemini


# %%
# --- Install Dependencies (if needed) ---
# !pip install -q google-genai python-dotenv

import os
import json
import time
import shutil
import logging
from functools import wraps
from pathlib import Path
from google import genai
from google.genai import types

# Try loading dotenv for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- API Key Setup ---
if os.environ.get("KAGGLE_CONTAINER_NAME"):
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    try:
        GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    except Exception:
        # Fallback or manual entry
        GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
else:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not found. Please set it in your secrets or environment.")

# Configure retry options for API resilience
retry_config = types.HttpRetryOptions(
    attempts=5,           # Maximum retry attempts
    exp_base=2,           # Exponential backoff multiplier
    initial_delay=2,      # Initial delay before first retry (seconds)
    http_status_codes=[429, 500, 503, 504]  # Retry on these HTTP errors
)

# Initialize Client with retry configuration
client = genai.Client(
    api_key=GOOGLE_API_KEY,
    http_options=types.HttpOptions(retry_options=retry_config)
)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
print(f"Using model: {GEMINI_MODEL}")

# %%
# --- Configuration Parameters ---
# Modify these variables to control the pipeline

# Input: Path to your audio file or text notes
# This is your stream of conciouness that you want to refine
INPUT_PATH = "data/inputs_kaggle/raw_thoughts/train_of_thought.m4a"

# Context: Path to context/instructions (optional)
# This is the format you want it generated. Eg: IM, email, speech; along with any other instructions.
CONTEXT_PATH = "data/inputs_kaggle/context.txt"

# Output: Where to save the final markdown
OUTPUT_PATH = "churchill_kaggle.md"

# Mode: Set True to skip style mimicry and just get a clean professional draft
CLEAN_ONLY = False

# Style Directory: Where to save/load the processed style profile
# Not relevant when using CLEAN_ONLY=True
STYLE_DIR = "my_style/"

# Style: Path to examples (folder of .txt/.md files) to mimic
# Not relevant when using CLEAN_ONLY=True
# Set to None to load from a previously populated STYLE_DIR
STYLE_EXAMPLES_PATH = "data/style_examples/"


# %%
# --- Sample Configurations for Demoing ---
# Uncomment one of these blocks to demo different scenarios
# --- Scenario 1: Clean-only mode (no style mimicry) ---
STYLE_EXAMPLES_PATH = None
STYLE_DIR = None
CLEAN_ONLY = True

# --- Scenario 2: Generate existing style profile from the developer ---
# STYLE_EXAMPLES_PATH = None
# STYLE_DIR = "my_warren_style/"
# CLEAN_ONLY = False

# --- Scenario 3: Generate new style profile from Winston Churchill ---
STYLE_EXAMPLES_PATH = "data/style_examples_churchill/"
STYLE_DIR = "my_churchill_style/"
CLEAN_ONLY = False


# %%
# --- Configuration you probably don't need to change ---
# Working Directories
WORKING_DIR = Path.cwd()
DEBUG_DIR = WORKING_DIR / "debug_output"

# Supported file extensions
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.ogg', '.flac'}
TEXT_EXTENSIONS = {'.txt', '.md'}

# %%
# --- Observability: Logging & Metrics ---
# Configure structured logging for debugging and production monitoring

# Set global log level to warn so Gemini API doesn't spam with HTTP info.
logging.basicConfig(
    level=logging.WARN,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)

# Locally, log level INFO will print summary but not live-tracing
logger = logging.getLogger("ghostwriter")
logger.setLevel(logging.INFO)

# Metrics collection for performance tracking
metrics = {
    "api_calls": 0,
    "total_duration_seconds": 0.0,
    "agent_durations": {},
    "hierarchical_durations": {},
    "iteration_scores": [],
    "errors": [],
    "token_usage": {
        "total": {"input": 0, "output": 0},
        "by_agent": {}
    }
}

# Global stack for context tracking so we can give a tree of performance info
_agent_stack = []

def _get_current_agent() -> str:
    """
    Get the name of the currently executing agent from the stack.

    Returns:
        str: The name of the current agent or "unknown" if the stack is empty.
    """
    return _agent_stack[-1] if _agent_stack else "unknown"


# Adding traced annotation automatically logs duration, stack level, and errors.
# Token tracking is done in the generate() wrapper.
def traced(func):
    """
    Decorator that adds observability to functions:
    - Logs function entry and exit
    - Tracks execution duration
    - Records metrics for performance analysis
    - Captures and logs errors

    This implements the observability pattern from the ADK course,
    providing visibility into the agent pipeline execution.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        _agent_stack.append(func_name)
        start_time = time.perf_counter()

        logger.debug(f"▶ START: {func_name}")

        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time

            # Record metrics
            metrics["agent_durations"][func_name] = metrics["agent_durations"].get(func_name, 0) + elapsed

            # Hierarchical tracking
            stack_tuple = tuple(_agent_stack)
            metrics["hierarchical_durations"][stack_tuple] = metrics["hierarchical_durations"].get(stack_tuple, 0) + elapsed

            # Only add to total if it's a root call to avoid double counting
            if len(_agent_stack) == 1:
                metrics["total_duration_seconds"] += elapsed

            logger.debug(f"✓ END: {func_name} ({elapsed:.2f}s)")
            return result

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            error_info = {"function": func_name, "error": str(e), "duration": elapsed}
            metrics["errors"].append(error_info)

            logger.error(f"✗ FAILED: {func_name} ({elapsed:.2f}s) - {e}")
            raise
        finally:
            if _agent_stack:
                _agent_stack.pop()

    return wrapper


def log_metrics_summary():
    """Logs a summary of collected metrics at the end of the pipeline."""
    logger.info("\n" + "="*50)
    logger.info("📊 PIPELINE METRICS SUMMARY")
    logger.info("="*50)
    logger.info(f"Total API Calls: {metrics['api_calls']}")
    logger.info(f"Total Duration: {metrics['total_duration_seconds']:.2f}s")

    # Token Usage
    total_tokens = metrics["token_usage"]["total"]
    logger.info(f"Total Tokens: {total_tokens['input'] + total_tokens['output']} "
                f"(Input: {total_tokens['input']}, Output: {total_tokens['output']})")

    if metrics["hierarchical_durations"]:
        logger.info("\nExecution Tree:")
        for path in sorted(metrics["hierarchical_durations"].keys()):
            indent = "  " * (len(path) - 1)
            icon = "└─" if len(path) > 1 else "•"
            logger.info(f"{indent}{icon} {path[-1]}: {metrics['hierarchical_durations'][path]:.2f}s")

    if metrics["agent_durations"]:
        logger.info("Aggregated Durations:")
        for agent, duration in sorted(metrics["agent_durations"].items(), key=lambda x: -x[1]):
            logger.info(f"  {agent}: {duration:.2f}s")

    if metrics["token_usage"]["by_agent"]:
        logger.info("Token Usage by Agent:")
        for agent, usage in sorted(metrics["token_usage"]["by_agent"].items(), key=lambda x: -(x[1]['input'] + x[1]['output'])):
            total = usage['input'] + usage['output']
            logger.info(f"  {agent}: {total} (In: {usage['input']}, Out: {usage['output']})")

    if metrics["iteration_scores"]:
        logger.info(f"\nIteration Scores: {metrics['iteration_scores']}")
        logger.info(f"Final Score: {metrics['iteration_scores'][-1]}")

    if metrics["errors"]:
        logger.warning(f"Errors Encountered: {len(metrics['errors'])}")
        for err in metrics["errors"]:
            logger.warning(f"  {err['function']}: {err['error']}")

    logger.info("="*50 + "\n")

# %%
# --- Helper Functions ---

def generate(contents, config=None):
    """
    Core wrapper for LLM calls with built-in retry handling.

    Retry logic is configured at the client level via HttpRetryOptions,
    which automatically handles transient API errors (429, 500, 503, 504)
    with exponential backoff. Always uses the globally configured GEMINI_MODEL.

    Parameters:
        contents: The content/prompt to send to the model.
        config: Optional GenerateContentConfig for response formatting.

    Returns:
        The generated content response from the API.
    """
    metrics["api_calls"] += 1
    logger.debug(f"API Call #{metrics['api_calls']} to {GEMINI_MODEL}")

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=config
    )

    # Capture tokens
    try:
        usage = response.usage_metadata
        if usage:
            input_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0

            metrics["token_usage"]["total"]["input"] += input_tokens
            metrics["token_usage"]["total"]["output"] += output_tokens

            # Attribute to current agent
            current_agent = _get_current_agent()
            if current_agent:
                if current_agent not in metrics["token_usage"]["by_agent"]:
                    metrics["token_usage"]["by_agent"][current_agent] = {"input": 0, "output": 0}
                metrics["token_usage"]["by_agent"][current_agent]["input"] += input_tokens
                metrics["token_usage"]["by_agent"][current_agent]["output"] += output_tokens
    except Exception as e:
        logger.warning(f"Failed to capture token metrics: {e}")

    return response


def generate_text(contents) -> str:
    """
    Simplified helper that returns just the text response.

    Use this for simple text generation where you don't need
    special config options like JSON output.

    Parameters:
        contents: The content/prompt to send to the model.

    Returns:
        The generated text as a string.
    """
    return generate(contents).text

def setup_debug_dir():
    """Cleans and recreates the debug output directory."""
    if DEBUG_DIR.exists():
        shutil.rmtree(DEBUG_DIR)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Debug directory initialized at: {DEBUG_DIR}")

def save_debug_file(filename, content):
    """Saves content to the debug directory."""
    try:
        file_path = DEBUG_DIR / filename
        with open(file_path, "w", encoding="utf-8") as f:
            if isinstance(content, (dict, list)):
                json.dump(content, f, indent=2)
            else:
                f.write(str(content))
    except Exception as e:
        print(f"Warning: Failed to save debug file {filename}: {e}")

# %%
# --- Tool: Transcriber ---

@traced
def upload_audio(file_path: str):
    """Uploads audio file to Gemini Files API."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    print(f"Uploading {path.name}...")
    uploaded_file = client.files.upload(file=path)

    # Wait for processing
    while uploaded_file.state.name == "PROCESSING":
        print("Processing audio file...")
        time.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)

    if uploaded_file.state.name != "ACTIVE":
        raise Exception(f"File upload failed with state: {uploaded_file.state.name}")

    print(f"File {path.name} is active.")
    return uploaded_file

@traced
def transcribe_audio(audio_file_path: str) -> str:
    """
    Transcribes audio using Gemini
    """
    try:
        audio_file = upload_audio(audio_file_path)

        prompt = """
        Transcribe this audio file verbatim.
        Include all hesitations, filler words (um, ah), and false starts.
        Do not summarize or edit. Just write exactly what is said.
        """

        response = generate(
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(
                            file_uri=audio_file.uri,
                            mime_type=audio_file.mime_type
                        ),
                        types.Part.from_text(text=prompt)
                    ]
                )
            ]
        )

        return response.text
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

# %%
# --- Tool: Style Architect ---

def load_style_examples(source_path: str) -> list[str]:
    """
    Reads all text files from the style examples directory or a single file.

    Parameters:
        source_path (str): The path to the file or directory containing style examples.

    Returns:
        list[str]: A list of strings, where each string is the content of a file.
    """
    examples = []
    path = Path(source_path)

    if not path.exists():
        print(f"Path {path} does not exist.")
        return []

    files_to_read = []
    if path.is_file():
        files_to_read = [path]
    elif path.is_dir():
        files_to_read = list(path.glob("*.md")) + list(path.glob("*.txt"))

    for file_path in files_to_read:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                examples.append(f.read())
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return examples

@traced
def generate_style_profile(examples: list[str]) -> tuple[str, list[str]] | tuple[None, None]:
    """
    Generates a style system prompt and snippets using Gemini.

    Parameters:
        examples (list[str]): A list of text examples to analyze.

    Returns:
        tuple[str, list[str]] | tuple[None, None]: A tuple containing the style prompt and a list of style snippets, or (None, None) if no examples are provided.
    """
    if not examples:
        print("No style examples found.")
        return None, None

    combined_text = "\n\n---\n\n".join(examples)

    # Check if combined text is too large for a single pass (approx 2M chars)
    MAX_CHUNK_SIZE = 2_000_000

    if len(combined_text) <= MAX_CHUNK_SIZE:
        # Standard single-pass processing
        return _generate_style_profile_single_pass(combined_text)
    else:
        # Multi-pass iterative processing
        print(f"⚠️ Style examples too large ({len(combined_text)} chars). Processing in chunks...")
        return _generate_style_profile_iterative(combined_text, MAX_CHUNK_SIZE)

def _generate_style_profile_single_pass(text_content: str) -> tuple[str, list[str]]:
    """
    Helper for standard single-pass style generation.

    Parameters:
        text_content (str): The combined text content of all style examples.

    Returns:
        tuple[str, list[str]]: A tuple containing the style prompt and a list of style snippets.
    """
    # 1. Generate Style System Prompt
    print("Analyzing writing style...")
    prompt_analysis = """
    Analyze the following text samples to create a comprehensive "Style System Prompt".
    This prompt will be used to instruct an AI to write exactly like this author.

    Focus on:
    - Sentence structure (length, complexity, rhythm)
    - Vocabulary (simple vs complex, jargon, specific words to avoid)
    - Tone (formal, casual, humorous, cynical, etc.)
    - Formatting preferences
    - Common rhetorical devices (analogies, metaphors, questions)

    Output ONLY the system prompt text. Do not include introductory text like "Here is the prompt".
    Start directly with "You are a writer who..." or similar instructions.
    """

    response_prompt = generate(
        contents=[prompt_analysis, text_content]
    )
    style_prompt = response_prompt.text.strip()

    # 2. Extract Style Snippets
    print("Extracting style snippets...")
    prompt_snippets = """
    Extract 5-10 distinct "Style Snippets" from the text samples.
    These should be representative paragraphs or sections that best showcase the author's unique voice.

    Output the result as a JSON list of strings.
    Example: ["Snippet 1 text...", "Snippet 2 text..."]
    """

    response_snippets = generate(
        contents=[prompt_snippets, text_content],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    try:
        style_snippets = json.loads(response_snippets.text)
    except json.JSONDecodeError:
        print("Error decoding snippets JSON. Saving raw text.")
        style_snippets = [response_snippets.text]

    return style_prompt, style_snippets

def _generate_style_profile_iterative(full_text: str, chunk_size: int) -> tuple[str, list[str]]:
    """
    Iteratively updates style profile across large datasets.

    Parameters:
        full_text (str): The full text content to process.
        chunk_size (int): The size of each chunk to process.

    Returns:
        tuple[str, list[str]]: A tuple containing the final style prompt and a list of style snippets.
    """
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

    current_style_prompt = "No style defined yet."
    all_snippets = []

    for i, chunk in enumerate(chunks):
        print(f"Processing style chunk {i+1}/{len(chunks)}...")

        # Update Prompt
        prompt_update = f"""
        You are building a comprehensive "Style System Prompt".

        CURRENT STYLE PROMPT:
        {current_style_prompt}

        NEW TEXT SAMPLES:
        {chunk[:50000]}... (truncated for brevity in prompt)

        Task: Update and refine the Current Style Prompt based on the New Text Samples.
        Ensure the new prompt captures nuances from BOTH the previous definition and the new samples.

        Output ONLY the updated system prompt.
        """
        response = generate(contents=[prompt_update])
        current_style_prompt = response.text.strip()

        # Extract Snippets from this chunk
        prompt_snippets = """
        Extract 3 distinct "Style Snippets" from these text samples that showcase unique voice.
        Output as JSON list of strings.
        """
        try:
            resp = generate(
                contents=[prompt_snippets, chunk],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            new_snippets = json.loads(resp.text)
            all_snippets.extend(new_snippets)
        except Exception as e:
            print(f"Warning: Failed to extract snippets from chunk {i+1}: {e}")

    # Cap snippets at 10 to avoid context bloat
    import random
    if len(all_snippets) > 10:
        all_snippets = random.sample(all_snippets, 10)

    return current_style_prompt, all_snippets

def process_style_examples(source_path: str) -> tuple[str, list[str]] | tuple[None, None]:
    """
    Process style examples from a path and return the profile.

    Parameters:
        source_path (str): The path to the style examples.

    Returns:
        tuple[str, list[str]] | tuple[None, None]: A tuple containing the style prompt and snippets, or (None, None) if no examples found.
    """
    examples = load_style_examples(source_path)
    if not examples:
        return None, None
    return generate_style_profile(examples)

# %%
# --- Tool: Context Engineer ---

@traced
def validate_context_length(text: str, max_chars: int = int(1e6) * 3):
    """
    Validates that the context length is within manageable limits.

    Instead of lossy compaction, we enforce a hard limit to ensure quality.
    Future versions will implement a 'Chunking Agent' to handle large contexts losslessly.

    Parameters:
        text: The input text to validate.
        max_chars: Character limit (approx 6000 tokens) before error triggers.

    Raises:
        ValueError: If text exceeds the limit.
    """
    if len(text) > max_chars:
        raise ValueError(
            f"Context length ({len(text)} chars) exceeds limit ({max_chars}). "
            "Automatic compaction is disabled to prevent data loss. "
            "Please split your input or implement a Chunking Agent."
        )
    return text

# %%
# --- Agents ---

@traced
def agent_listener(audio_path: str) -> str:
    """
    Agent 1: Transcribes audio.

    Parameters:
        audio_path (str): The path to the audio file.

    Returns:
        str: The transcribed text.
    """
    print("\n--- Agent 1: The Listener ---")
    print(f"Listening to {audio_path}...")
    transcript = transcribe_audio(audio_path)
    print("Transcription complete.")
    return transcript

@traced
def agent_editor(raw_transcript: str) -> str:
    """
    Agent 2: Cleans up the transcript.

    Parameters:
        raw_transcript (str): The raw transcript text.

    Returns:
        str: The cleaned draft text.
    """
    print("\n--- Agent 2: The Editor ---")
    print("Cleaning up transcript...")

    prompt = """
    You are an expert editor. Your task is to take a raw, verbatim transcript of a stream-of-consciousness recording and clean it up.

    1. Remove filler words (um, ah, like, you know).
    2. Fix false starts and stuttering.
    3. Organize the thoughts into a logical flow / outline.
    4. Do NOT rewrite the content in a new style yet; just make the content clear and coherent.

    Output the cleaned draft.
    """

    return generate_text([prompt, raw_transcript])

@traced
def agent_strategist(context_transcript: str) -> str:
    """
    Agent 1.5: The Strategist. Converts context transcript into a writing goal.

    Parameters:
        context_transcript (str): The transcript of the context/instructions.

    Returns:
        str: The generated writing brief.
    """
    print("\n--- Agent 1.5: The Strategist ---")
    print("Analyzing context to define writing goals...")

    prompt = """
    You are an expert content strategist. Your task is to analyze a raw transcript of instructions/context and distill it into a clear, actionable "Writing Brief".

    The Brief should include:
    1. **Goal**: What is the primary purpose of this piece?
    2. **Target Audience**: Who is this for?
    3. **Key Requirements**: Specific points that must be covered.
    4. **Tone/Style Direction**: Any specific instructions on how it should sound (e.g., "formal", "casual", "like a speech").
    5. **Length**: The desired length (e.g., "short email", "long report", "3 paragraphs"). If not specified, infer a reasonable length based on the goal.
    6. **Medium**: The delivery format (e.g., "Email", "Slack message", "Blog post", "Speech"). If not specified, infer it from context.

    Output the Brief in a clear, structured format.
    """

    return generate_text([prompt, context_transcript])

@traced
def agent_ghostwriter(cleaned_draft: str, style_prompt: str, style_snippets: list[str], writing_brief: str | None = None, critique: str | None = None) -> str:
    """
    Agent 3: Rewrites the draft in the user's style.

    Parameters:
        cleaned_draft (str): The cleaned draft text.
        style_prompt (str): The style system prompt.
        style_snippets (list[str]): List of style snippets.
        writing_brief (str | None, optional): The writing brief. Defaults to None.
        critique (str | None, optional): Critique from the previous iteration. Defaults to None.

    Returns:
        str: The rewritten draft.
    """
    print("\n--- Agent 3: The Ghostwriter ---")
    if critique:
        print("Applying critique to revision...")
    else:
        print("Drafting initial version...")

    # Context Engineering: Combine System Prompt + Few-Shot Snippets
    snippets_text = "\n\n".join([f"Example {i+1}:\n{s}" for i, s in enumerate(style_snippets)])

    full_prompt = f"""
    {style_prompt}

    Here are some examples of your past writing to guide your style:
    {snippets_text}
    """

    if writing_brief:
        full_prompt += f"\n\nWRITING BRIEF / GOALS:\n{writing_brief}\n"

    full_prompt += """
    Task: Rewrite the following draft in your unique style.
    """

    if critique:
        full_prompt += f"\n\nCRITICAL FEEDBACK FROM PREVIOUS DRAFT:\n{critique}\n\nPlease address this feedback in your rewrite."

    full_prompt += f"\n\nDRAFT TO REWRITE:\n{cleaned_draft}"

    return generate_text([full_prompt])

@traced
def agent_critic(original_transcript: str, generated_text: str, style_prompt: str, writing_brief: str | None = None, iteration: int = 1) -> dict:
    """
    Agent 4: Evaluates the output.

    Parameters:
        original_transcript (str): The original transcript.
        generated_text (str): The generated text to evaluate.
        style_prompt (str): The style rules.
        writing_brief (str | None, optional): The writing brief. Defaults to None.
        iteration (int, optional): The current iteration number. Defaults to 1.

    Returns:
        dict: A dictionary containing the score, critique, and pass status.
    """
    print("\n--- Agent 4: The Critic ---")
    print(f"Evaluating draft (Iteration {iteration})...")

    prompt = f"""
    You are a highly critical, hard-to-please literary editor and fact-checker. You rarely give perfect scores.

    Input 1: Original Transcript (Source of Truth)
    Input 2: Generated Text (Style Mimicry)
    Input 3: Style Rules
    Input 4: Writing Brief / Goals
    Input 5: Iteration Number: {iteration}

    Style Rules:
    {style_prompt}

    Writing Brief:
    {writing_brief if writing_brief else "None provided."}

    Task:
    1. FACTUAL ACCURACY: Verify that the Generated Text accurately reflects the content of the Original Transcript. Any hallucination or missing key detail is an automatic fail.
    2. STYLE ADHERENCE: Evaluate how well the Generated Text follows the Style Rules.
    3. GOAL ALIGNMENT: Most importantly of all: Verify that the text meets the goals, audience, tone, length, and medium defined in the Writing Brief.

    Scoring Criteria:
    - 10: Flawless. Indistinguishable from the target author, perfect accuracy, perfect brief alignment. (Extremely rare)
    - 9: Near Perfect. Maybe one tiny nitpick.
    - 8: Excellent. Publishable with no further edits.
    - 6-7: Good. Some stylistic slips, minor tone issues, or could be tighter. (REQUIRES REVISION)
    - 4-5: Mediocre. Missed the style or brief significantly.
    - <4: Poor. Factual errors or complete style mismatch.

    CRITICAL INSTRUCTIONS:
    1. If the Writing Brief specifies a length (e.g., "concise", "short", "3 paragraphs"), you MUST penalize the score if the text is verbose or overly long, even if the style is otherwise perfect.
    2. ITERATION 1 BIAS: If this is Iteration 1, be EXTREMELY SKEPTICAL. It is statistically improbable that a first draft is perfect. Look deeper. Is the rhythm perfect? Is the vocabulary varied enough? Is the tone exactly right? If there is ANY room for improvement, score it a 7 or lower to force a revision.

    Output JSON:
    {{
        "score": <int 1-10>,
        "critique": "<detailed explanation of what to improve, specifically referencing the style rules and brief>",
        "pass": <boolean, true if score >= 8>
    }}
    """

    user_content = f"""
    ORIGINAL TRANSCRIPT:
    {original_transcript}

    GENERATED TEXT:
    {generated_text}
    """

    response = generate(
        contents=[prompt, user_content],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    try:
        result = json.loads(response.text)
        # Ensure 'pass' key exists
        if 'pass' not in result:
            result['pass'] = result.get('score', 0) >= 8
        return result
    except Exception:
        return {"score": 0, "critique": "Error parsing critic response", "pass": False}

# %%
# --- Agent Dependent Helper Functions ---

@traced
def process_input_source(source_path: str) -> str:
    """
    Processes an input source (file or directory).

    Parameters:
        source_path (str): The path to the input source.

    Returns:
        str: The combined text content from the source.
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Input source not found: {source_path}")

    files_to_process = []
    if path.is_dir():
        files_to_process = [p for p in path.rglob("*") if p.is_file() and not p.name.startswith(".")]
        files_to_process.sort()
    else:
        files_to_process = [path]

    combined_text = []

    for file_path in files_to_process:
        print(f"Processing {file_path}...")
        suffix = file_path.suffix.lower()

        if suffix in AUDIO_EXTENSIONS:
            print(f"Transcribing audio: {file_path}")
            transcript = agent_listener(str(file_path))
            combined_text.append(f"--- Source: {file_path.name} ---\n{transcript}")
        elif suffix in TEXT_EXTENSIONS:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    combined_text.append(f"--- Source: {file_path.name} ---\n{content}")
            except Exception as e:
                print(f"Error reading text file {file_path}: {e}")
        else:
            print(f"Skipping unsupported file type: {file_path}")

    return "\n\n".join(combined_text)

@traced
def load_style_data_config(style_examples_path: str | None = None, style_dir: str | None = None) -> tuple[str, list[str]]:
    """
    Loads style data based on configuration.

    Parameters:
        style_examples_path (str | None, optional): Path to style examples. Defaults to None.
        style_dir (str | None, optional): Directory to save/load style data. Defaults to None.

    Returns:
        tuple[str, list[str]]: A tuple containing the style prompt and snippets.
    """

    # Make sure load/save directory exists
    if style_dir:
        save_dir = Path(style_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError("STYLE_DIR must be specified to save generated style profile.")

    # Scenario 1: Generate New Style
    if style_examples_path:
        print(f"Generating new style profile from: {style_examples_path}")
        style_prompt, style_snippets = process_style_examples(style_examples_path)

        if style_prompt and style_snippets:
            # Save to the specified output directory (or default)
            prompt_file = save_dir / "style_prompt.txt"
            snippets_file = save_dir / "style_snippets.json"

            print(f"Saving style profile to: {save_dir}")
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(style_prompt)
            with open(snippets_file, "w", encoding="utf-8") as f:
                json.dump(style_snippets, f, indent=2)
            return style_prompt, style_snippets
        else:
            raise ValueError("Failed to generate style profile from provided examples.")

    # Scenario 2: Load Existing Style
    style_dir_path = Path(style_dir)
    prompt_file = style_dir_path / "style_prompt.txt"
    snippets_file = style_dir_path / "style_snippets.json"

    if not prompt_file.exists() or not snippets_file.exists():
        raise FileNotFoundError(f"No style data found in {style_dir}. Please provide STYLE_EXAMPLES_PATH to generate one.")

    print(f"Loading style profile from: {style_dir}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        style_prompt = f.read()

    with open(snippets_file, "r", encoding="utf-8") as f:
        style_snippets = json.load(f)

    return style_prompt, style_snippets

# %%
# --- Run Pipeline ---
print("Starting The Ghostwriter...")
setup_debug_dir()

# 0. Setup Style
if CLEAN_ONLY:
    print("Clean-only mode: Using standard professional style.")
    style_prompt = "Style: Clear, professional, and concise standard English. Do not mimic any specific author."
    style_snippets = []
else:
    try:
        style_prompt, style_snippets = load_style_data_config(
            style_examples_path=STYLE_EXAMPLES_PATH,
            style_dir=STYLE_DIR
        )
    except Exception as e:
        print(f"Error loading style data: {e}")
        # Stop execution if style fails
        raise e

# 1. Get Content Input
try:
    raw_transcript = process_input_source(INPUT_PATH)
    if not raw_transcript:
        print("No content found in input.")
        raise ValueError("Empty input")
    save_debug_file("1_raw_transcript.txt", raw_transcript)
except Exception as e:
    print(f"Error processing input: {e}")
    raise e

# 2. Get Context Input
writing_brief = None
if CONTEXT_PATH:
    print(f"\nProcessing context: {CONTEXT_PATH}")
    try:
        context_transcript = process_input_source(CONTEXT_PATH)
        if context_transcript:
            # Validate Context Length (Context Engineering)
            validate_context_length(context_transcript)

            save_debug_file("2_context_transcript.txt", context_transcript)
            writing_brief = agent_strategist(context_transcript)
            save_debug_file("3_writing_brief.txt", writing_brief)
            print(f"\n=== WRITING BRIEF ===\n{writing_brief}\n=====================\n")
    except Exception as e:
        print(f"Error processing context: {e}")

# 3. Editor
cleaned_draft = agent_editor(raw_transcript)
save_debug_file("4_cleaned_draft.md", cleaned_draft)
print(f"Cleaned Draft: {cleaned_draft[:100]}...")

# 4. Ghostwriter & Critic Loop
max_retries = 3
current_try = 0
critique = None
final_text = None

while current_try < max_retries:
    current_try += 1
    print(f"\n--- Iteration {current_try}/{max_retries} ---")

    # Generate
    draft = agent_ghostwriter(cleaned_draft, style_prompt, style_snippets, writing_brief, critique)
    save_debug_file(f"5_draft_iteration_{current_try}.md", draft)
    print(f"Generated Draft (Snippet):\n{draft[:200]}...\n")

    # Evaluate
    evaluation = agent_critic(raw_transcript, draft, style_prompt, writing_brief, current_try)
    save_debug_file(f"5_critique_iteration_{current_try}.json", evaluation)
    # Track score in metrics
    metrics["iteration_scores"].append(evaluation['score'])

    print(f"Score: {evaluation['score']}/10")
    print(f"Critique: {evaluation['critique']}")

    if evaluation['pass']:
        print("\nSUCCESS! Draft approved.")
        final_text = draft
        break
    else:
        critique = evaluation['critique']

if not final_text:
    print("\nMax retries reached. Returning last draft.")
    final_text = draft

# Output
output_path = WORKING_DIR / OUTPUT_PATH
with open(output_path, "w", encoding="utf-8") as f:
    f.write(final_text)

print("\n--- Final Output ---")
print(final_text)

print(f"\nFinal piece saved to {output_path}")

# Log metrics summary
log_metrics_summary()
