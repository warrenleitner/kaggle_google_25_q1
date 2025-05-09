# %% [markdown]
# # GenAI Workout Coach

# %% [markdown]
# ## Background
# 
# ### The Problem
# I've tried a lot of fitness apps over the years, and I mean a LOT. And I've spent a good bit of money on it too. My favorite run training app is over \\$100/year. My favorite strength training app is \\$60/year. And just last month, Garmin launched an "AI" version of their Connect app at \\$7/month.
# 
# Obviously, there are free versions, but most of them use very fixed and not personalized routines. These paid apps are much better, but I find they aren't very flexible, aren't as tailored to the user as I would like, and really only serve their immediate goal: Running apps aren't very good for strength training and strength training apps aren't very good for running. It's not holistic and there's no way to add feedback or anything specific to you.
# 
# ### Inspiration
# A few months ago I started using Deepseek to plan my runs, and help specifically with some of my pesornal issues that even the best paid apps don't do well with: managing my shin pain, and dealing with my motivation issues. The drawback was that it was a bit of a pain to type in all of my data for each workout, and I wanted to make it easier.
# 
# ### The Goal
# My goal with this project, at least eventually, is that I can integrate it with Apple Health to retrieve my workout and biometric data automatically. Unfortunately that's a bit of a high bar for this project, but this is already significantly easier than my Deepseek method, via the use of structured output and multimodal input of workout screenshots or spoken descriptions of goals.

# %% [markdown]
# ## App
# 
# ### App Structure
# The overall design of this app is to gather data at various intervals and granularities, make sure that's all up to date, feed that into an LLM to generate a workout plan, let the user provide feedback to adjust the plan, then ask any questions related to the plan.
# 
# The core of the app are two GenAI functions, `call_gemini_for_structured_output` and `get_user_input_with_multimodal`. These are used at each of the data gathering phases to update the user's statistics. There's a wrapper function for each type of data, which generates a prompt for the user and another for the system.
# * `get_user_input_with_multimodal` is used initially with the user prompt to get some starting point of the data relevant at that phase. The user can provide text input, and upload whatever files they deem relevant. For demo purposes, I've used an audio recording for the initial user info, and screenshots for monthly and per-workout statistics. If the LLM decides that it needs additional information, this function is used again to collect it.
# * `call_gemini_for_structured_output` This consists of two AI agents. The first can solely call functions, and its goal is to gather more data until it has enough info to populate the applicable JSON schema. It either calls a function to describe what kind of info it needs, or one to indicate that it has enough info. The second one solely outputs the JSON format requested.
# 
# After all of the info has been updated, it is saved in JSON files so the user doesn't have to start from scratch every time. Then the app continues to the planning phase, where another agent is fed ALL of the JSON data gathered above, and generates a workout plan in a Markdown format. The output isn't formally structured, but follows the format of an assessment of the last workout, progress towards the goal, an overview of the next week of workouts, details of the very next workout (plus a fallback if the user doesn't like the main option), and some motivation. The user can either accept the plan or provide feedback, which is fed back into the LLM for refinement.
# 
# Once the user accepts the plan, the same data, plus the generated plan, is fed into a final LLM, which does not edit the plan but acts as an advisor, where the user can ask questions about their data or the plan. As an example, I have used it to ask for running form cues.
# 
# ### Demonstrated Techniques
# The app explicitly generates 4 of the required techniques, and implicitly uses several more.
# Explicit:
# * **Structured output/JSON mode/controlled generation** - Used specifically by `call_gemini_for_structured_output` to generate the JSON that describes each type of data.
# * **Image understanding** - Can be uplodaed for any of the info gathering phases, but is demonstrated specifically for the monthly and per-workout data.
# * **Audio understanding** - Can be uplodaed for any of the info gathering phases, but is demonstrated specifically for the one-time user info.
# * **Function Calling** - Used by `call_gemini_for_structured_output` to force the model to only either prompt the user for more data or proceed to JSON generation.
# 
# Implicit:
# * **Agents** - Arguable, but I would offer that the specialized LLM usage for different phases of the app demonstrate agentic LLMs
# * **Long context window** - A basic input to the Q&A phase in my sample run consumes approx 20k tokens. More back and forth or more data would consume more, not to mention the file parsing.
# * **Context caching** While not explicitly controllable, the iterative input in data gathering phases should allow the LLM to use context caching in its processing. Since I already had the required minimum techniques, I didn't check specifically to see if this was happening.
# 
# Not shown, but **Document understanding** was used at an earlier phase to parse a LOT of workouts into a "baseline" JSON so I had more data to work with. This could eventually be worked into an "import" feature, but for now it was just used one time, and the prompt changed back to working with a single workout at once.
# 
# ## Status
# 
# ### Approach
# At first I was hoping that by writing a thorough enough description, I could coerce some LLM to create a usable prototype for me. I started with my goal, and talked it over and refined it with Gemini Pro. Then I started talking with it about implementation, and told it that one of my goals was to do this in about 8 hours. I forget its initial estimate but it was something outrageous like 120 hours. Thankfully, telling it I didn't care about an interface or quality 😅 cut it significantly. But it also helped me scale back my goals equally, and I'm glad it did. I didn't track it, but I ended up spending about twice that long on my project.
# 
# A lot of that time was on two fool's errands. First, I was way too optimisti about having the LLM write most or all of it for me. Ironically, it seems that LLMs are not nearly as good at writing LLM code as more generic code, which, I suppose, shouldn't be too surprising considering the sample size is so much smaller. In the end, I had to more or less throw away the version that the LLM generated for me.
# 
# The second wild goose chase was about agent mode. In part, it's tightly linked to the first. It seems that LLM interfaces are changing rapidly, because it was not accurate at all in the APIs it tried to use or how it tried to use them. In the end, I stumbled upon the function calling barista example linked from the agent barista example, and that was a godsend, because that was almost exactly what I needed for my main loop.
# 
# From there, it gradual refinement and bug fixing of the major issues. I apologize for the non-Kaggle-ness of the style. I developed it as a local script and then broke it into code cells.
# 
# ### Current State
# I'd consider the current quality rough POC. Most runs (pun-intended 😆) need some massaging, but it is usable enough for me to actually try it for my daily workouts. You can try itusing the dataset at [kaggle.com/warrenleitner/kaggle-google-25-q1-data](http://kaggle.com/warrenleitner/kaggle-google-25-q1-data)
# 
# ### Future Enhancements
# In the future, there's a lot I can do to improve this. To start with, I'll just use it and see how useful it is, but I'm already envisioning next steps:
# 1. Obviously make it more robust in terms of error handling, input/output validation, pruning old data, automatically aggregating metrics, etc.
# 2. Integrate with Apple Health so all quantitative data is updated automatically. Even if I just had a super simple app or even a Siri shortcut that output a CSV of the info I need, that would be perfect. In fact it might be easier and more consistent to always generate the JSON this way, but it's very convenient to be able to upload data in any format or even as audio/images to the LLM.
# 3. Automatic LLM based generation of the JSON schema so it can flex even more to the user's preferences and always collect the most relevant data.
# 4. Some structured saving of the workout plan, so that the LLM can do a more detailed assessment of the user's performance vs the planned workout, and "track" that data over time.
# 5. .fit or other standard workout format generation, so that workouts can be automatically laoded into training apps, rather than needing to create them manually or attempt to follow them even more manually.
# 

# %% [markdown]
# ## Helpers

# %% [markdown]
# ### Generic Imports

# %%
# Kaggle special: pip commands
#!pip install google-generativeai rich

# %%
from google import genai
import google.ai.generativelanguage as glm
from google.api_core import retry
from google.generativeai.types import ContentDict
import os
import json
from datetime import datetime, date, timedelta
import typing
import time
from pathlib import Path

import sys
from datetime import date
import sys

# Use typing.TypedDict for Python >= 3.12, otherwise use typing_extensions.TypedDict
if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


# %% [markdown]
# ### Local vs Kaggle Environment Setup

# %%
# Detects whether running in Kaggle or local environment and sets up API keys accordingly
if os.environ.get("KAGGLE_CONTAINER_NAME"):
    # Running in Kaggle - use Kaggle's secrets management
    WORKING_DIR = Path("/kaggle/working/")
    DATA_DIR = Path("/kaggle/input/")
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
        print("Successfully configured Gemini API Key from Kaggle Secrets.")
    except Exception as e:
        print(f"Error accessing Kaggle secrets or configuring API key: {e}")
        print("Please ensure GOOGLE_API_KEY is added as a Kaggle secret.")
        raise
else:
    # Running locally - use environment variables
    WORKING_DIR = Path.cwd() / "logs"
    DATA_DIR = Path.cwd() / "data"
    if os.environ.get("GOOGLE_API_KEY"):
        GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
        print("Successfully configured Gemini API Key from environment variable.")
    else:
        print("No API key found. Please set GOOGLE_API_KEY in your environment variables.")
        raise ValueError("No API key found. Please set GOOGLE_API_KEY in your environment variables.")

# %% [markdown]
# ### Constants

# %%
# Define paths to all persistent data files
USER_DATA_FILE = WORKING_DIR / "user_data.json"                # Static user profile information
DAILY_FEELING_FILE = WORKING_DIR / "daily_feelings.json"       # Daily subjective wellness tracking
WORKOUT_DATA_FILE = WORKING_DIR / "workout_data.json"          # Individual workout session data
WEEKLY_SUMMARY_FILE = WORKING_DIR / "weekly_summaries.json"    # Weekly aggregated summaries
MONTHLY_STATS_FILE = WORKING_DIR / "monthly_stats.json"        # Monthly fitness statistics
WORKOUT_PLAN_FILE = WORKING_DIR / "current_workout_plan.json"  # Latest workout plan

# %% [markdown]
# ### Initial Gemini Configuration

# %% [markdown]
# Borrow retry method from the labs

# %%
# Initialize the client with API key and configure retry logic for resilience
client = genai.Client(api_key=GOOGLE_API_KEY)

# Define which error types are retriable (rate limits and service unavailable)
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

# Apply retry decorator to the generate_content method if not already wrapped
if not hasattr(genai.models.Models.generate_content, '__wrapped__'):
  genai.models.Models.generate_content = retry.Retry(
      predicate=is_retriable)(genai.models.Models.generate_content)

# %% [markdown]
# ### JSON Helper Functions

# %% [markdown]
# Boilerplate data loading/saving/initialization functions. One interesting one that I dealt with was dates. Surprisingly, at least to me, the LLM was only able to generate datetimes. Not dates. So lots of stuff here to deal with that because for this app, times were too granular for what the user should be thinking about.

# %%
class DateEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles datetime and date objects.
    
    Converts datetime and date objects to ISO format strings during JSON serialization.
    This ensures proper saving of temporal data to JSON files.
    """
    def default(self, obj) -> typing.Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date) and not isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def normalize_date_to_datetime(date_value) -> typing.Optional[datetime]:
    """
    Converts various date formats to datetime object at midnight.
    
    Args:
        date_value: Can be a datetime object, date object, or string in ISO format
                   or YYYY-MM-DD format
    
    Returns:
        A datetime object set to midnight of the given date, or None if conversion fails
    """
    if date_value is None:
        return None
    if isinstance(date_value, datetime):
        return date_value
    elif isinstance(date_value, date) and not isinstance(date_value, datetime):
        # Convert date to datetime at midnight
        return datetime.combine(date_value, datetime.min.time())
    elif isinstance(date_value, str):
        try:
            # Handle ISO format with time component
            if 'T' in date_value:
                return datetime.fromisoformat(date_value)
            else:
                # Handle simple date format without time
                return datetime.strptime(date_value, '%Y-%m-%d')
        except ValueError:
            pass
    return None

def date_decoder(json_dict: dict) -> dict:
    """
    Object hook for json.loads() that converts date strings to datetime objects.
    
    Processes a dictionary during JSON deserialization and converts string values
    in date-related fields to datetime objects.
    
    Args:
        json_dict: Dictionary being processed during JSON loading
        
    Returns:
        The modified dictionary with string dates converted to datetime objects
    """
    for key, value in json_dict.items():
        if isinstance(value, str) and key in ['date', 'goal_date', 'week_start_date']:
            try:
                # First try ISO format with time (YYYY-MM-DDThh:mm:ss)
                if 'T' in value:
                    json_dict[key] = datetime.fromisoformat(value)
                else:
                    # Try simple date format (YYYY-MM-DD) - set time to midnight
                    json_dict[key] = datetime.strptime(value, '%Y-%m-%d')
            except ValueError:
                pass # Keep original string if not a valid date
    return json_dict

def load_json_data(filepath: Path) -> typing.Union[dict, list, None]:
    """
    Loads JSON data from a file, handling potential errors.
    
    Includes special handling for datetime fields to ensure proper deserialization
    of date values from JSON strings to Python datetime objects.
    
    Args:
        filepath: Path object pointing to the JSON file to load
        
    Returns:
        Dictionary or list parsed from the JSON file, or None if the file doesn't exist or has errors
    """
    print(f"Loading data from {filepath.name}...")
    if not filepath.exists():
        print(f"File {filepath.name} does not exist. Will initialize later.")
        return None
    try:
        print(f"Reading and parsing {filepath.name}...")
        with open(filepath, 'r') as f:
            # Use object_hook for date decoding
            data = json.load(f, object_hook=date_decoder)
            
            # Extra processing for lists of dicts (to handle nested date fields)
            if isinstance(data, list):
                print(f"Processing datetime fields in list data ({len(data)} items)...")
                for item in data:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if isinstance(value, str) and key in ['date', 'goal_date', 'week_start_date']:
                                try:
                                    # Handle ISO format with time
                                    if 'T' in value:
                                        item[key] = datetime.fromisoformat(value)
                                    else:
                                        item[key] = datetime.strptime(value, '%Y-%m-%d')
                                except ValueError:
                                    pass
            print(f"Successfully loaded data from {filepath.name}")
            return data
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading JSON from {filepath}: {e}")
        return None

def save_json_data(filepath: Path, data: typing.Union[dict, list]) -> None:
    """
    Saves data to a JSON file.
    
    Handles creating parent directories if needed and uses the custom DateEncoder
    to properly serialize datetime objects to ISO format strings.
    
    Args:
        filepath: Path object pointing to where the JSON file should be saved
        data: Dictionary or list to serialize to JSON
    """
    print(f"Saving data to {filepath.name}...")
    try:
        # Ensure directory exists before saving
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            # Use cls=DateEncoder for date encoding
            json.dump(data, f, indent=4, cls=DateEncoder)
        print(f"Data successfully saved to {filepath.name}")
    except IOError as e:
        print(f"Error saving JSON to {filepath}: {e}")
    except TypeError as e:
        print(f"Error serializing data for {filepath}: {e}")


def initialize_data_files() -> None:
    """
    Creates empty files if they don't exist (except user data).
    
    Ensures that all necessary collection files exist as empty arrays
    to avoid errors when the application tries to read from them later.
    The user data file is handled separately since it requires input.
    """
    print("Initializing data files for first-time use...")
    if not DAILY_FEELING_FILE.exists():
        print(f"Creating empty daily feelings file at {DAILY_FEELING_FILE.name}...")
        save_json_data(DAILY_FEELING_FILE, [])
    if not WORKOUT_DATA_FILE.exists():
        print(f"Creating empty workout data file at {WORKOUT_DATA_FILE.name}...")
        save_json_data(WORKOUT_DATA_FILE, [])
    if not WEEKLY_SUMMARY_FILE.exists():
        print(f"Creating empty weekly summary file at {WEEKLY_SUMMARY_FILE.name}...")
        save_json_data(WEEKLY_SUMMARY_FILE, [])
    if not MONTHLY_STATS_FILE.exists():
        print(f"Creating empty monthly stats file at {MONTHLY_STATS_FILE.name}...")
        save_json_data(MONTHLY_STATS_FILE, [])
    print("Data file initialization complete.")

def check_last_updated(filepath: Path) -> typing.Optional[datetime]:
    """
    Gets the last modified time of a file.
    
    Args:
        filepath: Path object pointing to the file to check
        
    Returns:
        Datetime object representing when the file was last modified, or None if the file
        doesn't exist or there's an error accessing it
    """
    if not filepath.exists():
        return None
    try:
        return datetime.fromtimestamp(filepath.stat().st_mtime)
    except OSError:
        return None

# %% [markdown]
# ### GenAI Utility Functions

# %% [markdown]
# Now we're getting into the interesting parts. This first function handles the Gemini Files API to allow multi-modal input, and the one below is a prompting loop that handles either text input or a list of file inputs. The third one in this block formats them for the `content` array argument to the `generate_content` function.

# %%
def upload_file_to_gemini(filepath: Path) -> typing.Optional[glm.File]:
    """
    Uploads a local file to the Gemini Files API.
    
    Handles checking if the file exists, uploading it to Gemini, and waiting
    for the file to become active in Gemini's system.
    
    Args:
        filepath: Path object pointing to the file to upload
        
    Returns:
        A Gemini File object that can be used in API calls, or None if upload fails
    """
    print(f"Attempting to upload {filepath.name} to Gemini Files API...")
    try:
        # Ensure the file exists before attempting upload
        if not filepath.is_file():
            print(f"Error: File not found at {filepath}")
            return None

        # The Gemini API client handles the upload process
        print(f"Sending file {filepath.name} to Gemini API...")
        uploaded_file = client.files.upload(file=filepath)
        print(f"Upload initiated for {filepath.name} as {uploaded_file.name}")

        # Check file state and wait for it to become active if needed
        print(f"Checking file state for {uploaded_file.name}...")
        file_info = client.files.get(name=uploaded_file.name)
        if file_info.state.name != "ACTIVE":
             print(f"File {uploaded_file.name} not yet active, current state: {file_info.state.name}")
             print(f"Waiting for file {uploaded_file.name} to become active...")
             # Wait loop - check every 2 seconds for up to 10 seconds total
             for i in range(5):
                 print(f"Waiting attempt {i+1}/5...")
                 time.sleep(2)
                 file_info = client.files.get(name=uploaded_file.name)
                 if file_info.state.name == "ACTIVE":
                     print(f"File {uploaded_file.name} is now active!")
                     break
             if file_info.state.name != "ACTIVE":
                 print(f"Warning: File {uploaded_file.name} did not become active quickly. Current state: {file_info.state.name}")
                 raise Exception(f"Warning: File {uploaded_file.name} did not become active quickly. State: {file_info.state.name}")

        print(f"Successfully uploaded {filepath.name} as {uploaded_file.name}")
        return uploaded_file # Return the File object needed for generate_content

    except Exception as e:
        print(f"Error uploading file {filepath.name} to Gemini: {e}")
        raise

# --- User Input and API Interaction Functions ---
def get_user_input_with_multimodal(prompt_text: str) -> typing.Tuple[str, typing.List[glm.File]]:
    """
    Gets text input and potentially file paths for multimodal input.
    
    Prompts the user for text responses and optional file uploads, which
    can then be processed by the Gemini model to extract information.
    
    Args:
        prompt_text: The question or instruction to display to the user
        
    Returns:
        A tuple of (user_text, files) where user_text is a string and files is a
        list of uploaded Gemini File objects. Returns empty lists if user skips.
    """
    print(f"\n{prompt_text}")
    user_text = input("Your text response (press enter to proceed to file uploads or type 'skip' to ignore this update): ")
    if user_text.lower() == 'skip':
        return []

    files = []
    while True:
        file_path_str = input("Enter path to an image/video/audio file to upload (Press Enter with no path to finish): ")
        if not file_path_str:
            break

        # Try multiple path resolution strategies to find the file
        # First check data directory
        if os.path.exists(DATA_DIR / file_path_str):
            file_path = DATA_DIR / file_path_str
        # Then check Kaggle working directory
        elif os.path.exists(Path("/kaggle") / file_path_str):
            file_path = Path.cwd() / file_path_str
        # Then check current directory
        elif os.path.exists(Path.cwd() / file_path_str):
            file_path = Path.cwd() / file_path_str
        # Finally try as absolute path
        elif os.path.exists(file_path_str):
            file_path = Path(file_path_str)
        else:
            print(f"Error: Could not find file at '{file_path_str}'.")
            continue  # Ask for another file path

        try:
            # Upload the file to Gemini API
            uploaded_file = client.files.upload(file=file_path)
            files.append(uploaded_file)
            print(f"Added {file_path.name} to input.")
        except Exception as e:
            print(f"Error uploading file {file_path.name} to Gemini: {e}")
            raise

    return user_text, files

def parse_input_into_messages(user_text: str, user_files: typing.List[glm.File]) -> typing.List[ContentDict]:
    """
    Prepares the initial message list for the first Gemini call.
    
    Formats user text and file inputs into the ContentDict structure required
    by the Gemini API for multi-part messages.
    
    Args:
        user_text: The text input from the user
        user_files: List of Gemini File objects that have been uploaded
        
    Returns:
        A list of ContentDict objects ready to be passed to the Gemini API
    """
    # Start with the text message
    messages = [
        ContentDict(role="user", parts=[genai.types.Part.from_text(text=user_text)])
    ]
    
    # Add each file as a separate message
    for item in user_files:
        messages.append(
            ContentDict(role="user", parts=[genai.types.Part.from_uri(file_uri=item.uri, mime_type=item.mime_type)])
        )
    return messages
    

# %% [markdown]
# ### Function Calling Targets
# 
# Here are our two function-calling targets. `request_more_user_info` is an actually useful function, and allows the LLM to use the `get_user_input_with_multimodal` function. `proceed_to_json_generation` is more just a stub to give it a choice so we have a binary sort of "good/not good".

# %%
# --- Function Calling for Information Gathering ---
def request_more_user_info(
    missing_fields: typing.List[str],  # Which fields are unclear or missing
    clarification_needed: str  # Specific question to ask the user
) -> typing.Optional[typing.List[ContentDict]]:
    """
    Called by the Gemini model when user input lacks required information.
    
    This function is exposed to the Gemini API as a tool that the model can call
    when it determines more information is needed from the user. It displays the
    model's specific questions to the user and collects their response.
    
    Args:
        missing_fields: A list of field names that the model determined were 
                       missing or unclear from the user's input
        clarification_needed: The specific question the model wants to ask the user
                             to obtain the missing information
                             
    Returns:
        A list of ContentDict messages containing the user's response, or None
        if the user provided no further input
    """
    # Format a clear prompt for the user
    prompt = "\n--- AI Needs More Information ---"
    prompt += f"\nMissing or unclear fields: {', '.join(missing_fields)}"
    prompt += f"\nAI Request: {clarification_needed}"
    prompt += "\n---------------------------------"
    
    # Get user response
    new_user_text, new_user_files = get_user_input_with_multimodal(prompt)
    
    # Format response for the API
    if new_user_text or new_user_files:
            return parse_input_into_messages(new_user_text, new_user_files)
    else:
        # User provided no further input
        print("User provided no further input for clarification.")
    
    return None

def proceed_to_json_generation() -> bool:
    """
    Called by the Gemini model when it determines it has sufficient information.
    
    This function is exposed to the Gemini API as a tool that the model can call
    when it has all the information it needs to generate structured JSON output.
    It signals to the calling code that the information gathering phase is complete.
    
    Returns:
        True, to signal that the information gathering loop should end
    """
    return True
    

# %% [markdown]
# ### Pièce de Résistance
# 
# This function is the real centerpoint for the app. All of hte data gathering goes through here. As described in the intro, in consists of an initial prompt to get some baseline data, a function-calling based, LLM driven refinement loop, and then the JSON structured output.

# %%
def call_gemini_for_structured_output(
    system_prompt: str,
    user_text: str,
    user_files: typing.List[glm.File],
    output_schema: typing.Type[TypedDict],
) -> typing.Dict[str, typing.Any]:
    """
    Calls Gemini with function calling enabled to gather data and generate structured JSON.
    
    This function implements an interactive loop with the Gemini API:
    1. First phase: The model can request more information from the user using function calling
    2. Second phase: Once sufficient information is gathered, generate structured JSON output
    
    Args:
        system_prompt: Instructions for the Gemini model about what data to collect
        user_text: Initial text input from the user
        user_files: List of uploaded file objects from the user
        output_schema: The TypedDict class defining the expected output structure
        
    Returns:
        A dictionary matching the output_schema structure with data extracted from user input
    """
    # Parse user input into message format
    print(f"Preparing messages for Gemini API call to extract {output_schema.__name__}...")
    messages = parse_input_into_messages(user_text, user_files)
    sufficient_info = False

    # Set the model to use
    model_name = "gemini-2.0-flash"
    print(f"Using model: {model_name}")

    # Information gathering loop - runs until sufficient info or max loops reached
    max_loops = 3
    loop_count = 0
    while ((not sufficient_info) and (loop_count < max_loops)):
        loop_count += 1
        print(f"Calling Gemini to gather data (loop {loop_count}/{max_loops})...")
        
        try:
            # Call Gemini with function calling enabled
            print("Sending request to Gemini with function calling enabled...")
            response = client.models.generate_content(
                model=model_name,
                config=genai.types.GenerateContentConfig(
                    # Define the tools (functions) the model can call
                    tools=[request_more_user_info, proceed_to_json_generation],
                    tool_config=genai.types.ToolConfig(
                        function_calling_config=genai.types.FunctionCallingConfig(
                            mode="ANY", allowed_function_names=["request_more_user_info", "proceed_to_json_generation"]
                        )
                    ),
                    system_instruction=system_prompt,
                    automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(disable=True)
                ),
                contents=messages
            )
            print("Received response from Gemini API")

            # Process the function call response
            tool_call = response.candidates[0].content.parts[0].function_call
            print(f"Gemini requested function call: {tool_call.name}")
            messages.append(genai.types.ContentDict(role="user", parts=[genai.types.Part.from_text(text=f"NOTE: System replied with - Function call: {tool_call.name} with args: {tool_call.args}")]))
            
            # Handle different function calls
            try:

                if tool_call.name == "proceed_to_json_generation":
                    print(f"Gemini has sufficient information to generate structured output.")
                    sufficient_info = proceed_to_json_generation()
                elif tool_call.name == "request_more_user_info":
                    print(f"Gemini needs more information from the user.")
                    more_info_messages = request_more_user_info(**tool_call.args)
                    if more_info_messages: # Check if the function returned a list
                        print(f"Adding user's additional information to the conversation.")
                        messages.extend(more_info_messages)
                    else:
                        print("User provided no additional information.")
                else:
                    print(f"Unknown tool call: {tool_call.name}")
                    messages.append(genai.types.ContentDict(role="user", parts=[genai.types.Part.from_text(text=f"NOTE: System function call INVALID: {tool_call.name}")]))
            except Exception as e:
                print(f"Error in Gemini API call: {e}")
                messages.append(genai.types.ContentDict(role="user", parts=[genai.types.Part.from_text(text=f"NOTE: System function call INVALID: {tool_call.name}\n\nError: {e}")]))
        except Exception as e:
            print(f"Error in Gemini API call: {e}")
            print(f"Messages: {messages}")
            print(f"Response: {response}")
            raise

    # Once we have sufficient information, generate structured JSON output
    print(f"\nAttempting to generate structured JSON for {output_schema.__name__}...")
    
    print("Sending request to Gemini for structured output generation...")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=genai.types.GenerateContentConfig(
            temperature=0.2, # Lower temp for more deterministic JSON extraction
            response_mime_type="application/json",
            response_schema=output_schema
        ),
        contents=messages
    )
    print("Received structured output response from Gemini")

    # --- Process Final Response ---
    try:
        if response.candidates and response.candidates[0].content.parts:
             # Get the JSON part from the response
             final_part = response.candidates[0].content.parts[0]
             if final_part.text: # Check if text part exists (where JSON is expected)
                 # Parse the JSON string into a Python dictionary
                 print("Parsing JSON response...")
                 json_string = final_part.text
                 extracted_data = json.loads(json_string, object_hook=date_decoder)

                 # Validate the extracted data is a dictionary
                 if isinstance(extracted_data, dict):
                     print(f"Successfully extracted structured data for {output_schema.__name__}.")
                     return extracted_data
                 else:
                     print(f"Error: Gemini response was not a valid JSON object. Type: {type(extracted_data)}")
                     raise Exception(f"Error: Gemini response was not a valid JSON object for {output_schema.__name__}. Response: {extracted_data}")
                     
             else:
                  print(f"Error: Response part did not contain text content.")
                  raise Exception(f"Error: Gemini response part did not contain text for {output_schema.__name__}.")

        else:
            print(f"Error: No valid candidates or content parts in response.")
            raise Exception(f"Error: No valid candidates or content parts in Gemini response for {output_schema.__name__}.")

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        raise Exception(f"Error decoding JSON response from Gemini for {output_schema.__name__}: {e}\n\nRaw response text: {response.candidates[0].content.parts[0].text}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise Exception(f"Unexpected error processing final Gemini response: {e}\n\nFull Response: {response}")

# %% [markdown]
# ### One Time Init Data

# %% [markdown]
# Now we're into the data gathering phase. This is a big section but it follows a very similar format. There's a section for each type of data, static, monthly, weekly, daily, and per workout. 
# 
# Each one consists of a TypedDict class definition that represents the JSON info, a JSON schema detailed description that can be used to guide both the data gathering and structured output LLMs, then a wrapper function that defines the user and system prompts, calls the primary LLM function, and saves the data. It also presents the results in a user readable format.

# %%
# --- 1. Static User Data ---
class StaticUserData(TypedDict):
    """
    Stores baseline user information relevant for training plan personalization.
    
    This represents relatively stable profile information that doesn't change frequently,
    including personal details, fitness goals and training preferences.
    """
    name: str  # User's name
    age: int  # User's age in years
    height_inches: typing.Optional[float] # User's height in inches (optional)
    gender: typing.Optional[str] # User's reported gender (optional, e.g., "Male", "Female", "Non-binary", "Prefer not to say")
     # --- Goal --- 
    goal: str  # User's primary goal (e.g., "5K", "10K", "Half Marathon", "Marathon", "Ultra", "Other")
    goal_time: typing.Optional[str]  # Target time for the goal (e.g., 2.5 for 2 hours 30 minutes)
    goal_date: typing.Optional[datetime]  # Date of the user would like to be ready for their goal. (e.g., "2024-05-15")
     # --- Training Preferences (Optional) ---
    preferred_training_days: typing.List[str] # Days user prefers to train (e.g., ["Monday", "Wednesday", "Friday", "Sunday"])
    long_run_day: typing.Optional[str] # Preferred day for the weekly long run (e.g., "Saturday")
     # --- Other Optional Data ---
    notes: typing.Optional[str]  # Optional free-text notes for additional context

# JSON schema definition for StaticUserData - used for Gemini API structured output validation
StaticUserDataSchema = """
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "StaticUserData",
  "description": "Stores baseline user information relevant for training plan personalization.",
  "type": "object",
  "properties": {
    "name": {
      "description": "User's name",
      "type": ["string", "null"]
    },
    "age": {
      "description": "User's age in years",
      "type": ["integer", "null"],
      "minimum": 0
    },
    "height_inches": {
      "description": "User's height in inches (optional)",
      "type": ["number", "null"],
       "exclusiveMinimum": 0
    },
    "gender": {
      "description": "User's reported gender (optional, e.g., \"Male\", \"Female\", \"Non-binary\", \"Prefer not to say\")",
      "type": ["string", "null"]
    },
    "goal": {
      "description": "User's primary goal (e.g., \"5K\", \"10K\", \"Half Marathon\", \"Marathon\", \"Ultra\", \"Other\")",
      "type": ["string", "null"]
    },
    "goal_time": {
      "description": "Target time for the goal (e.g., \"sub 2:30\" or 2.5 for 2 hours 30 minutes)",
      "type": ["string", "null"]
    },
    "goal_date": {
      "description": "Date user would like to be ready for their goal (e.g., \"2024-09-15\")",
       "type": ["string", "null"],
       "format": "date-time"
    },
    "preferred_training_days": {
      "description": "Days user prefers to train (e.g., [\"Monday\", \"Wednesday\", \"Friday\", \"Sunday\"])",
      "type": ["array", "null"],
      "items": {
        "type": "string"
      }
    },
    "long_run_day": {
      "description": "Preferred day for the weekly long run (e.g., \"Saturday\")",
      "type": ["string", "null"]
    },
    "notes": {
      "description": "Optional free-text notes for additional context",
      "type": ["string", "null"]
    }
  },
   "required": ["name", "age", "goal"]
}
"""

# --- Data Collection Functions ---
# These functions prompt users for different types of data and use Gemini API
# to extract structured information from their responses.

def collect_static_user_data() -> typing.Optional[StaticUserData]:
    """
    Collects initial user profile information if it doesn't exist.
    
    Prompts the user for their basic information including personal details,
    fitness goals, and training preferences. This data serves as the foundation
    for personalized training plan creation.
    
    Returns:
        A StaticUserData dictionary with the collected profile information,
        or None if collection was skipped or failed
    """
    print("\n--- Collecting Initial User Information ---")
    print("This process will gather your basic profile and fitness goals.")
    
    # System prompt instructs the AI how to collect user data
    system_prompt = f"""
    You are an AI fitness coach assistant. Your task is to collect essential information
    from the user to personalize their fitness plan. Ask clarifying questions if needed,
    but primarily focus on gathering the data for the required JSON format.
    Ensure dates are in YYYY-MM-DD format.
    The required output format is JSON matching this structure: {StaticUserDataSchema}
    """
    
    # User-facing prompt explaining what information to provide
    user_prompt = (
        "Welcome! Let's set up your fitness profile. Please tell me about yourself "
        "and your fitness goals. You can provide details like:\n"
        "- Your age, height (inches), gender (optional)\n"
        "- Your main fitness goal (e.g., run a 5k, build muscle, general fitness)\n"
        "- If you have a target time or date for your goal (e.g., 'sub 25 minutes', 'by 2025-09-15')\n"
        "- Preferred training days (e.g., Monday, Wednesday, Friday)\n"
        "- Preferred day for a longer workout/run (if applicable)\n"
        "- Any other relevant notes (injuries, preferences, etc.)\n"
        "You can also upload relevant files (like past training logs, goal descriptions)."
    )
    
    # Get user input
    print("Waiting for user to input profile information...")
    user_text, user_files = get_user_input_with_multimodal(user_prompt)
    if not user_text and not user_files:
        print("User skipped initial data collection. Cannot proceed without basic profile.")
        return None

    print(f"Received user input: {len(user_text) if user_text else 0} characters of text and {len(user_files) if user_files else 0} files")
    
    # Use Gemini to extract structured data
    print("Processing user input to extract profile information...")
    extracted_data = call_gemini_for_structured_output(
        system_prompt=system_prompt,
        user_text=user_text,
        user_files=user_files,
        output_schema=StaticUserData,
    )

    # Return the extracted data with type validation
    if extracted_data:
        print("Successfully extracted user profile data.")
        
        # Add human-readable summary
        print("\n=== User Profile Summary ===")
        print(f"Name: {extracted_data.get('name')}")
        print(f"Age: {extracted_data.get('age')}")
        if extracted_data.get('height_inches') is not None:
            print(f"Height: {extracted_data.get('height_inches')} inches")
        if extracted_data.get('gender') is not None:
            print(f"Gender: {extracted_data.get('gender')}")
        print(f"Goal: {extracted_data.get('goal')}")
        if extracted_data.get('goal_time') is not None:
            print(f"Goal Time: {extracted_data.get('goal_time')}")
        if extracted_data.get('goal_date') is not None:
            print(f"Goal Date: {extracted_data.get('goal_date').date().isoformat() if isinstance(extracted_data.get('goal_date'), datetime) else extracted_data.get('goal_date')}")
        if extracted_data.get('preferred_training_days'):
            print(f"Preferred Training Days: {', '.join(extracted_data.get('preferred_training_days'))}")
        if extracted_data.get('long_run_day') is not None:
            print(f"Long Run Day: {extracted_data.get('long_run_day')}")
        if extracted_data.get('notes'):
            print(f"Notes: {extracted_data.get('notes')}")
        print("===========================")
        input("Press Enter to continue...")
        
        return typing.cast(StaticUserData, extracted_data)
    else:
        print("Failed to extract user data from input.")
        raise Exception("Could not extract user data.")

# %% [markdown]
# ### Workout Info

# %%
# --- 3. Workout Data ---
class WorkoutData(TypedDict):
    """
    Data for a single workout session (running or other types).
    
    Captures detailed information about each completed workout including objective 
    measurements (duration, distance, pace) as well as subjective assessments
    (perceived exertion, pain levels). This data forms the foundation for tracking
    training load and progress over time.
    """
    date: datetime  # Date of the workout
    workout_type: str  # E.g., "Easy Run", "Tempo", "Strength", "Yoga", "Cycling", "Other"
    perceived_exertion: int  # RPE scale 1-10 (Rate of Perceived Exertion)
    shin_and_knee_pain: int  # 0 (None) to 5 (Severe) (Specific to this user's shin pain)
    shin_tightness: int  # 0 (None) to 5 (Severe) (Specific to this user's shin tightness)
    workout_adherence: int # Percent adherence to the planned workout (0-100%)
    actual_duration_minutes: float # Total workout time in minutes
    actual_distance_miles: typing.Optional[float]  # Distance in miles (None if not applicable)
    average_pace_minutes_per_mile: typing.Optional[float]  # Pace in min/mile (None if not applicable)
    average_heart_rate_bpm: typing.Optional[int]  # Average heart rate during workout if tracked
    max_heart_rate_bpm: typing.Optional[int]  # Maximum heart rate during workout if tracked
    elevation_gain_feet: typing.Optional[float]  # Elevation gain in feet (None if not applicable)
    notes: typing.Optional[str]  # User comments, could include exercises/sets/reps for strength

# JSON schema for WorkoutData
WorkoutDataSchema = """
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "WorkoutData",
  "description": "Data for a single workout session (running or other types).",
  "type": "object",
  "properties": {
    "date": {
      "description": "Date of the workout",
      "type": "string",
      "format": "date-time"
    },
    "workout_type": {
      "description": "E.g., \"Easy Run\", \"Tempo\", \"Strength\", \"Yoga\", \"Cycling\", \"Other\"",
      "type": "string"
    },
    "perceived_exertion": {
      "description": "RPE scale 1-10 (Made mandatory as it's often key)",
      "type": "integer",
      "minimum": 1,
      "maximum": 10
    },
    "shin_and_knee_pain": {
      "description": "0 (None) to 5 (Severe) (Specific to this user's shin pain)",
      "type": "integer",
      "minimum": 0,
      "maximum": 5
    },
    "shin_tightness": {
      "description": "0 (None) to 5 (Severe) (Specific to this user's shin tightness)",
      "type": "integer",
      "minimum": 0,
      "maximum": 5
    },
    "workout_adherence": {
      "description": "Percent adherence to the planned workout (0-100%)",
      "type": "integer",
      "minimum": 0,
      "maximum": 100
    },
    "actual_duration_minutes": {
      "description": "Total workout time in minutes",
      "type": "number",
       "exclusiveMinimum": 0
    },
    "actual_distance_miles": {
      "description": "Distance in miles (None if not applicable)",
      "type": ["number", "null"],
       "minimum": 0
    },
    "average_pace_minutes_per_mile": {
      "description": "Pace in min/mile (None if not applicable)",
      "type": ["number", "null"],
       "minimum": 0
    },
    "average_heart_rate_bpm": {
      "description": "Average Heart Rate (BPM) if tracked",
      "type": ["integer", "null"],
      "minimum": 0
    },
    "max_heart_rate_bpm": {
      "description": "Max Heart Rate (BPM) if tracked",
      "type": ["integer", "null"],
      "minimum": 0
    },
    "elevation_gain_feet": {
      "description": "Elevation gain in feet (None if not applicable)",
      "type": ["number", "null"]
    },
    "notes": {
      "description": "User comments, could include exercises/sets/reps for strength",
      "type": ["string", "null"]
    }
  },
  "required": [
    "date",
    "workout_type",
    "perceived_exertion",
    "shin_and_knee_pain",
    "shin_tightness",
    "workout_adherence",
    "actual_duration_minutes"
  ]
}
"""


# %% [markdown]
# ### Daily Info

# %%
# --- 2. Daily User Feeling ---
class DailyUserFeeling(TypedDict):
    """
    Subjective user input regarding their physical and mental state on a specific day.
    
    Captures how the user feels each day, focusing on metrics relevant to training readiness
    including energy levels, pain indicators, sleep quality, and other wellness factors.
    These daily entries help the coach adapt training plans based on recovery status.
    """
    date: datetime  # The date this feeling log applies to
    overall_feeling: int  # Rating scale, e.g., 1 (Very Poor) to 5 (Excellent)
    energy_level: int  # Rating scale, e.g., 1 (Very Low) to 5 (Very High)
    shin_pain: int  # Rating scale, e.g., 0 (None) to 5 (Severe)
    sleep_quality: int  # Rating scale, e.g., 1 (Very Poor) to 5 (Excellent)
    stress_level: typing.Optional[int]  # Optional: 1 (Very Low) to 5 (Very High)
    hydration_level: typing.Optional[int]  # 1 (Poor) to 5 (Excellent). Note that this represents the previous day
    nutrition_quality: typing.Optional[int]  # 1 (Poor) to 5 (Excellent). Note that this represents the previous day
    notes: typing.Optional[str]  # Optional free-text notes

# JSON schema for DailyUserFeeling
DailyUserFeelingSchema = """
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "DailyUserFeeling",
  "description": "Subjective user input regarding their physical and mental state on a specific day.",
  "type": "object",
  "properties": {
    "date": {
      "description": "The date this feeling log applies to",
      "type": "string",
      "format": "date-time"
    },
    "overall_feeling": {
      "description": "Rating scale, e.g., 1 (Very Poor) to 5 (Excellent)",
      "type": "integer",
      "minimum": 1,
      "maximum": 5
    },
    "energy_level": {
      "description": "Rating scale, e.g., 1 (Very Low) to 5 (Very High)",
      "type": "integer",
      "minimum": 1,
      "maximum": 5
    },
    "shin_pain": {
      "description": "Rating scale, e.g., 0 (None) to 5 (Severe)",
      "type": "integer",
      "minimum": 0,
      "maximum": 5
    },
    "sleep_quality": {
      "description": "Rating scale, e.g., 1 (Very Poor) to 5 (Excellent)",
      "type": "integer",
      "minimum": 1,
      "maximum": 5
    },
    "stress_level": {
      "description": "Optional: 1 (Very Low) to 5 (Very High)",
      "type": ["integer", "null"],
      "minimum": 1,
      "maximum": 5
    },
    "hydration_level": {
      "description": "1 (Poor) to 5 (Excellent). Note that this represents the previous day",
      "type": ["integer", "null"],
      "minimum": 1,
      "maximum": 5
    },
    "nutrition_quality": {
      "description": "1 (Poor) to 5 (Excellent). Note that this represents the previous day",
      "type": ["integer", "null"],
      "minimum": 1,
      "maximum": 5
    },
    "notes": {
      "description": "Optional free-text notes",
      "type": ["string", "null"]
    }
  },
  "required": [
    "date",
    "overall_feeling",
    "energy_level",
    "shin_pain",
    "sleep_quality"
  ]
}
"""



def collect_daily_feeling(target_date: datetime) -> typing.Optional[DailyUserFeeling]:
    """
    Prompts user for their daily subjective feeling input.
    
    Collects wellness data for a specific date including energy, pain levels,
    sleep quality, and other metrics that help assess recovery and readiness.
    
    Args:
        target_date: The date for which feeling data is being collected
        
    Returns:
        A DailyUserFeeling dictionary with the collected data,
        or None if collection was skipped or failed
    """
    print(f"\n--- Logging Daily Feeling for {target_date.date().isoformat()} ---")
    print("This will capture your subjective wellness metrics for today.")
    
    # System prompt for the AI
    system_prompt = f"""
    You are an AI fitness coach assistant. Collect the user's subjective feeling for {target_date.date().isoformat()}.
    Focus on the ratings (1-5 or 0-5 scales as defined) and any notes.
    The required output format is JSON matching this structure: {DailyUserFeelingSchema}
    Set the 'date' field to {target_date.isoformat()}.
    Remind the user that hydration and nutrition reflect the *previous* day.
    """
    
    # User-facing prompt explaining what to provide
    user_prompt = (
        f"How are you feeling today ({target_date.date().isoformat()})?\n"
        "- Overall feeling (1=Very Poor to 5=Excellent)?\n"
        "- Energy level (1=Very Low to 5=Very High)?\n"
        "- Shin pain (0=None to 5=Severe)?\n"
        "- Sleep quality last night (1=Very Poor to 5=Excellent)?\n"
        "- Stress level (Optional, 1=Very Low to 5=Very High)?\n"
        "- Hydration yesterday (Optional, 1=Poor to 5=Excellent)?\n"
        "- Nutrition yesterday (Optional, 1=Poor to 5=Excellent)?\n"
        "- Any other notes about how you feel?"
    )
    
    # Get user input
    print("Waiting for user to input daily feeling data...")
    user_text, user_files = get_user_input_with_multimodal(user_prompt)
    if not user_text and not user_files:
        print("User skipped daily feeling log.")
        return None
    
    print(f"Received user input: {len(user_text) if user_text else 0} characters of text and {len(user_files) if user_files else 0} files")
    
    # Use Gemini to extract structured data
    print("Processing user input to extract daily feeling data...")
    extracted_data = call_gemini_for_structured_output(
        system_prompt=system_prompt,
        user_text=user_text,
        user_files=user_files,
        output_schema=DailyUserFeeling,
    )

    # Return the extracted data with type validation
    if extracted_data:
        print("Successfully extracted daily feeling data.")
        
        # Add human-readable summary
        print("\n=== Daily Feeling Summary ===")
        print(f"Date: {target_date.date().isoformat()}")
        print(f"Overall Feeling: {extracted_data.get('overall_feeling')}/5")
        print(f"Energy Level: {extracted_data.get('energy_level')}/5")
        print(f"Shin Pain: {extracted_data.get('shin_pain')}/5")
        print(f"Sleep Quality: {extracted_data.get('sleep_quality')}/5")
        if extracted_data.get('stress_level') is not None:
            print(f"Stress Level: {extracted_data.get('stress_level')}/5")
        if extracted_data.get('hydration_level') is not None:
            print(f"Hydration Level: {extracted_data.get('hydration_level')}/5")
        if extracted_data.get('nutrition_quality') is not None:
            print(f"Nutrition Quality: {extracted_data.get('nutrition_quality')}/5")
        if extracted_data.get('notes'):
            print(f"Notes: {extracted_data.get('notes')}")
        print("============================")
        input("Press Enter to continue...")
        
        return typing.cast(DailyUserFeeling, extracted_data)
    else:
        print("Failed to extract daily feeling data from input.")
        raise Exception("Could not extract daily feeling data.")


# %% [markdown]
# ### Monthly Data

# %%
# --- 5. Monthly User Stats ---
class MonthlyUserStats(TypedDict):
    """
    Aggregated user statistics for a specific month.
    
    Focuses on quantitative metrics and fitness indicators tracked monthly,
    including physiological estimates (weight, heart rate, VO2 max) and 
    performance metrics (distances, paces). These statistics help track
    long-term fitness changes that might not be apparent in weekly data.
    """
    month: str  # Format: "YYYY-MM"
    # --- Physiological Measurements ---
    weight_pounds: float # User's current weight in pounds
    max_heart_rate_bpm: int # Max HR (estimated or tested)
    resting_heart_rate_bpm: int # Resting HR
    vo2_max_estimated: float # Estimated VO2 Max
    # --- Performance Metrics ---
    longest_run_distance_miles: float  # Longest single run distance this month
    average_pace_minutes_per_mile: float # Avg pace for runs this month
    comfortable_pace_minutes_per_mile: float # Comfortable pace for runs this month
    comfortable_run_distance_miles: float # Avg comfortable run distance
    average_heart_rate_bpm: int # Avg HR across workouts with HR data
    total_elevation_gain_feet: float # Total elevation for runs
    monthly_summary_notes: str # Field for brief LLM-generated or user notes

# JSON schema for MonthlyUserStats
MonthlyUserStatsSchema = """
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "MonthlyUserStats",
  "description": "Aggregated user statistics for a specific month.",
  "type": "object",
  "properties": {
    "month": {
      "description": "Format: \"YYYY-MM\"",
      "type": "string",
      "pattern": "^\\d{4}-\\d{2}$"
    },
    "weight_pounds": {
      "description": "User's current weight in pounds",
      "type": "number",
      "exclusiveMinimum": 0
    },
    "max_heart_rate_bpm": {
      "description": "Max HR (estimated or tested)",
      "type": "integer",
      "minimum": 0
    },
    "resting_heart_rate_bpm": {
      "description": "Resting HR",
      "type": "integer",
      "minimum": 0
    },
    "vo2_max_estimated": {
      "description": "Estimated VO2 Max",
      "type": "number",
      "minimum": 0
    },
    "longest_run_distance_miles": {
      "description": "Longest run distance this month",
      "type": "number",
      "minimum": 0
    },
    "average_pace_minutes_per_mile": {
      "description": "Avg pace for runs this month",
      "type": "number",
      "minimum": 0
    },
    "comfortable_pace_minutes_per_mile": {
      "description": "Comfortable pace for runs this month",
      "type": "number",
      "minimum": 0
    },
    "comfortable_run_distance_miles": {
      "description": "Avg comfortable run distance",
      "type": "number",
      "minimum": 0
    },
    "average_heart_rate_bpm": {
      "description": "Avg HR across workouts with HR data",
      "type": "integer",
      "minimum": 0
    },
    "total_elevation_gain_feet": {
      "description": "Total elevation for runs",
      "type": "number"
    },
    "monthly_summary_notes": {
      "description": "Field for brief LLM-generated or user notes",
      "type": "string"
    }
  },
  "required": [
    "month",
    "weight_pounds",
    "max_heart_rate_bpm",
    "resting_heart_rate_bpm",
    "vo2_max_estimated",
    "longest_run_distance_miles",
    "average_pace_minutes_per_mile",
    "comfortable_pace_minutes_per_mile",
    "comfortable_run_distance_miles",
    "average_heart_rate_bpm",
  ]
}
"""

def collect_monthly_stats(year: int, month: int, user_data: StaticUserData, workouts: typing.List[WorkoutData]) -> typing.Optional[MonthlyUserStats]:
    """
    Collects or generates monthly statistics for the user.
    
    Aggregates workout data for a specific month and combines it with user input
    to create a monthly snapshot of fitness metrics and indicators.
    
    Args:
        year: The year for the monthly stats
        month: The month number (1-12) for the stats
        user_data: Static user profile data
        workouts: List of workout data for filtering to the relevant month
        
    Returns:
        A MonthlyUserStats dictionary with the generated statistics,
        or None if collection was skipped or failed
    """
    print(f"\n--- Logging Monthly Statistics for {year}-{month:02d} ---")
    print("This will track key fitness metrics at a monthly level.")
    
    # Calculate the start date of the month
    month_start = datetime.combine(date(year, month, 1), datetime.min.time())
    
    # Calculate the end date of the month (inclusive)
    next_month_start = date(year, month + 1, 1) if month < 12 else date(year + 1, 1, 1)
    month_end = datetime.combine(next_month_start - timedelta(days=1), datetime.max.time())
    
    print(f"Analyzing data from {month_start.date().isoformat()} to {month_end.date().isoformat()}")
    
    # Filter workouts data to just this month
    month_workouts = [w for w in workouts if 
                     isinstance(w.get('date'), datetime) and 
                     month_start <= w.get('date') < datetime.combine(next_month_start, datetime.min.time())]
    
    print(f"Found {len(month_workouts)} workouts for the month")
    
    # Calculate aggregated metrics
    print("Calculating aggregated metrics for month...")
    
    # Extract running workouts
    run_workouts = [w for w in month_workouts 
                   if any(run_type in w.get('workout_type', '').lower() 
                         for run_type in ['run', 'tempo', 'interval', 'speed', 'jog'])]
    
    # Basic metrics
    total_workouts = len(month_workouts)
    running_workouts = len(run_workouts)
    
    # Find longest run
    longest_run = 0
    if run_workouts:
        longest_run = max((w.get('actual_distance_miles', 0) or 0) for w in run_workouts)
    
    # Calculate average pace for runs with distance data
    pace_runs = [w for w in run_workouts 
                if w.get('actual_distance_miles') and w.get('actual_duration_minutes')]
    
    avg_pace = None
    comfortable_pace = None
    if pace_runs:
        # Calculate weighted average pace
        total_miles = sum(w.get('actual_distance_miles', 0) or 0 for w in pace_runs)
        if total_miles > 0:
            total_mins = sum((w.get('actual_duration_minutes', 0) or 0) for w in pace_runs)
            avg_pace = total_mins / total_miles
            
            # Find reasonable average for "comfortable" pace
            # This is a rough estimate - use "easy" runs as proxy
            easy_runs = [r for r in pace_runs if 'easy' in r.get('workout_type', '').lower()]
            if easy_runs:
                easy_miles = sum(w.get('actual_distance_miles', 0) or 0 for w in easy_runs)
                if easy_miles > 0:
                    easy_mins = sum((w.get('actual_duration_minutes', 0) or 0) for w in easy_runs)
                    comfortable_pace = easy_mins / easy_miles
            
            # If no easy runs, estimate comfortable pace as slightly slower than average
            if comfortable_pace is None and avg_pace is not None:
                comfortable_pace = avg_pace * 1.1  # 10% slower than overall average
    
    # Calculate average run distance
    avg_run_distance = None
    if run_workouts:
        distance_runs = [r.get('actual_distance_miles', 0) or 0 for r in run_workouts]
        if any(d > 0 for d in distance_runs):
            avg_run_distance = sum(distance_runs) / len([d for d in distance_runs if d > 0])
    
    # Calculate HR metrics if available
    avg_hr = None
    hr_workouts = [w for w in month_workouts if w.get('average_heart_rate_bpm')]
    if hr_workouts:
        hr_values = [w.get('average_heart_rate_bpm', 0) for w in hr_workouts]
        avg_hr = sum(hr_values) / len(hr_values)
    
    # Calculate elevation gain
    total_elevation = 0
    for w in month_workouts:
        if w.get('elevation_gain_feet'):
            total_elevation += w.get('elevation_gain_feet', 0) or 0
    
    # Format summary data to display to user
    summary_text = (
        f"\nSummary for {year}-{month:02d}:\n"
        f"Total workouts: {total_workouts}\n"
        f"Running workouts: {running_workouts}\n"
        f"Longest run: {longest_run:.1f} miles\n"
    )
    
    if avg_pace:
        pace_min = int(avg_pace)
        pace_sec = int((avg_pace - pace_min) * 60)
        summary_text += f"Average pace: {pace_min}:{pace_sec:02d} min/mile\n"
    
    if comfortable_pace:
        c_pace_min = int(comfortable_pace)
        c_pace_sec = int((comfortable_pace - c_pace_min) * 60)
        summary_text += f"Comfortable pace: {c_pace_min}:{c_pace_sec:02d} min/mile\n"
    
    if avg_run_distance:
        summary_text += f"Average run distance: {avg_run_distance:.1f} miles\n"
    
    if avg_hr:
        summary_text += f"Average heart rate: {int(avg_hr)} bpm\n"
    
    if total_elevation > 0:
        summary_text += f"Total elevation gain: {total_elevation:.0f} feet\n"
    
    print(summary_text)
    
    # System prompt for the AI
    system_prompt = f"""
        You are an AI fitness coach assistant. Create monthly statistics for {year}-{month:02d} based on workout logs.
        Focus on key fitness metrics and indicators.
        The required output format is JSON matching this structure: {MonthlyUserStatsSchema}
        Set the 'month' field to "{year}-{month:02d}"
    """
    
    # Human-facing prompt for additional context
    user_prompt = (
        f"Let's capture your key stats for {year}-{month:02d}.\n\n"
        f"{summary_text}\n"
        "Please provide these required measurements:\n"
        "1. Current weight (in pounds)\n"
        "2. Resting heart rate (in bpm)\n"
        "3. Max heart rate (in bpm) - estimate with 220-age if unsure\n"
        "4. VO2max (estimated based on pace/HR data if available)\n\n"
        "Any other notes about your fitness this month? The AI will also generate a summary."
    )
    
    # Get user input for additional context
    print("Waiting for user to provide additional measurements...")
    user_text, user_files = get_user_input_with_multimodal(user_prompt)
    print(f"Received user input: {len(user_text) if user_text else 0} characters of text and {len(user_files) if user_files else 0} files")
    
    # Combine data and user input into a comprehensive context for the AI
    full_context = (
        f"Context for {year}-{month:02d} monthly statistics:\n"
        f"User Profile: {json.dumps(user_data, cls=DateEncoder, indent=2)}\n\n"
        f"Workout Details: {json.dumps(month_workouts, cls=DateEncoder, indent=2)}\n\n"
        f"Calculated statistics:\n"
        f"- Total workouts: {total_workouts}\n"
        f"- Running workouts: {running_workouts}\n"
        f"- Longest run: {longest_run:.1f} miles\n"
    )
    
    if avg_pace:
        full_context += f"- Average pace: {avg_pace:.2f} minutes per mile\n"
    
    if comfortable_pace:
        full_context += f"- Comfortable pace: {comfortable_pace:.2f} minutes per mile\n"
    
    if avg_run_distance:
        full_context += f"- Average run distance: {avg_run_distance:.2f} miles\n"
    
    if avg_hr:
        full_context += f"- Average heart rate: {int(avg_hr)} bpm\n"
    
    if total_elevation > 0:
        full_context += f"- Total elevation gain: {total_elevation:.0f} feet\n"
    
    full_context += "\nUser's input with measurements and context:"
    if user_text:
        full_context += "\n" + user_text
    
    # Use Gemini to generate the monthly stats
    print("Processing data to generate monthly statistics...")
    extracted_data = call_gemini_for_structured_output(
        system_prompt=system_prompt,
        user_text=full_context,
        user_files=user_files,
        output_schema=MonthlyUserStats,
    )
    
    # Process, validate and return the extracted data
    if extracted_data:
        print("Successfully generated monthly statistics.")
        
        # Ensure required fields are set
        extracted_data['month'] = f"{year}-{month:02d}"
        
        # Ensure fields have reasonable values
        if longest_run > 0:
            extracted_data['longest_run_distance_miles'] = longest_run
        
        if avg_pace:
            extracted_data['average_pace_minutes_per_mile'] = avg_pace
        
        if comfortable_pace:
            extracted_data['comfortable_pace_minutes_per_mile'] = comfortable_pace
        
        if avg_run_distance:
            extracted_data['comfortable_run_distance_miles'] = avg_run_distance
        
        if avg_hr:
            extracted_data['average_heart_rate_bpm'] = int(avg_hr)
        
        if total_elevation > 0:
            extracted_data['total_elevation_gain_feet'] = total_elevation
        
        # Add human-readable summary
        print("\n=== Monthly Stats ===")
        print(f"Month: {extracted_data.get('month')}")
        print(f"Weight: {extracted_data.get('weight_pounds'):.1f} lbs")
        print(f"Resting HR: {extracted_data.get('resting_heart_rate_bpm')} bpm")
        print(f"Max HR: {extracted_data.get('max_heart_rate_bpm')} bpm")
        print(f"Estimated VO2 Max: {extracted_data.get('vo2_max_estimated'):.1f}")
        print(f"Longest Run: {extracted_data.get('longest_run_distance_miles'):.1f} miles")
        if extracted_data.get('average_pace_minutes_per_mile'):
            avg_pace = extracted_data.get('average_pace_minutes_per_mile')
            avg_min = int(avg_pace)
            avg_sec = int((avg_pace - avg_min) * 60)
            print(f"Average Pace: {avg_min}:{avg_sec:02d} min/mile")
        if extracted_data.get('comfortable_pace_minutes_per_mile'):
            com_pace = extracted_data.get('comfortable_pace_minutes_per_mile')
            com_min = int(com_pace)
            com_sec = int((com_pace - com_min) * 60)
            print(f"Comfortable Pace: {com_min}:{com_sec:02d} min/mile")
        if extracted_data.get('comfortable_run_distance_miles'):
            print(f"Comfortable Run Distance: {extracted_data.get('comfortable_run_distance_miles'):.1f} miles")
        if extracted_data.get('average_heart_rate_bpm'):
            print(f"Average Heart Rate: {extracted_data.get('average_heart_rate_bpm')} bpm")
        if extracted_data.get('total_elevation_gain_feet'):
            print(f"Total Elevation Gain: {extracted_data.get('total_elevation_gain_feet'):.0f} feet")
        if extracted_data.get('monthly_summary_notes'):
            print(f"\nSummary Notes:\n{extracted_data.get('monthly_summary_notes')}")
        print("====================")
        input("Press Enter to continue...")
        
        return typing.cast(MonthlyUserStats, extracted_data)
    else:
        print("Failed to generate monthly statistics.")
        return None

# %% [markdown]
# ### Weekly Info

# %%
print("Gemini client and retry policy initialized successfully.")

# --- Data Structure Definitions ---
# The following classes define the structured data types used throughout the application.
# Each class uses TypedDict for strong typing and includes a corresponding JSON schema
# that defines validation rules and field descriptions.

# --- 1. Static User Data ---
class StaticUserData(typing.TypedDict):
    """
    Stores baseline user information relevant for training plan personalization.
    
    This represents relatively stable profile information that doesn't change frequently,
    including personal details, fitness goals and training preferences.
    """
    name: str  # User's name
    age: int  # User's age in years
    height_inches: typing.Optional[float] # User's height in inches (optional)
    gender: typing.Optional[str] # User's reported gender (optional, e.g., "Male", "Female", "Non-binary", "Prefer not to say")
     # --- Goal --- 
    goal: str  # User's primary goal (e.g., "5K", "10K", "Half Marathon", "Marathon", "Ultra", "Other")
    goal_time: typing.Optional[str]  # Target time for the goal (e.g., 2.5 for 2 hours 30 minutes)
    goal_date: typing.Optional[datetime]  # Date of the user would like to be ready for their goal. (e.g., "2024-05-15")
     # --- Training Preferences (Optional) ---
    preferred_training_days: typing.Optional[typing.List[str]] # Days user prefers to train (e.g., ["Monday", "Wednesday", "Friday", "Sunday"])
    long_run_day: typing.Optional[str] # Preferred day for the weekly long run (e.g., "Saturday")
     # --- Other Optional Data ---
    notes: typing.Optional[str]  # Optional free-text notes for additional context

# JSON schema definition for StaticUserData - used for Gemini API structured output validation
StaticUserDataSchema = """
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "StaticUserData",
  "description": "Stores baseline user information relevant for training plan personalization.",
  "type": "object",
  "properties": {
    "name": {
      "description": "User's name",
      "type": ["string", "null"]
    },
    "age": {
      "description": "User's age in years",
      "type": ["integer", "null"],
      "minimum": 0
    },
    "height_inches": {
      "description": "User's height in inches (optional)",
      "type": ["number", "null"],
       "exclusiveMinimum": 0
    },
    "gender": {
      "description": "User's reported gender (optional, e.g., \"Male\", \"Female\", \"Non-binary\", \"Prefer not to say\")",
      "type": ["string", "null"]
    },
    "goal": {
      "description": "User's primary goal (e.g., \"5K\", \"10K\", \"Half Marathon\", \"Marathon\", \"Ultra\", \"Other\")",
      "type": ["string", "null"]
    },
    "goal_time": {
      "description": "Target time for the goal (e.g., \"sub 2:30\" or 2.5 for 2 hours 30 minutes)",
      "type": ["string", "null"]
    },
    "goal_date": {
      "description": "Date user would like to be ready for their goal (e.g., \"2024-09-15\")",
       "type": ["string", "null"],
       "format": "date-time"
    },
    "preferred_training_days": {
      "description": "Days user prefers to train (e.g., [\"Monday\", \"Wednesday\", \"Friday\", \"Sunday\"])",
      "type": ["array", "null"],
      "items": {
        "type": "string"
      }
    },
    "long_run_day": {
      "description": "Preferred day for the weekly long run (e.g., \"Saturday\")",
      "type": ["string", "null"]
    },
    "notes": {
      "description": "Optional free-text notes for additional context",
      "type": ["string", "null"]
    }
  },
   "required": ["name", "age", "goal"]
}
"""

# --- 2. Daily User Feeling ---
class DailyUserFeeling(typing.TypedDict):
    """
    Subjective user input regarding their physical and mental state on a specific day.
    
    Captures how the user feels each day, focusing on metrics relevant to training readiness
    including energy levels, pain indicators, sleep quality, and other wellness factors.
    These daily entries help the coach adapt training plans based on recovery status.
    """
    date: datetime  # The date this feeling log applies to
    overall_feeling: int  # Rating scale, e.g., 1 (Very Poor) to 5 (Excellent)
    energy_level: int  # Rating scale, e.g., 1 (Very Low) to 5 (Very High)
    shin_pain: int  # Rating scale, e.g., 0 (None) to 5 (Severe)
    sleep_quality: int  # Rating scale, e.g., 1 (Very Poor) to 5 (Excellent)
    stress_level: typing.Optional[int]  # Optional: 1 (Very Low) to 5 (Very High)
    hydration_level: typing.Optional[int]  # 1 (Poor) to 5 (Excellent). Note that this represents the previous day
    nutrition_quality: typing.Optional[int]  # 1 (Poor) to 5 (Excellent). Note that this represents the previous day
    notes: typing.Optional[str]  # Optional free-text notes

# JSON schema for DailyUserFeeling
DailyUserFeelingSchema = """
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "DailyUserFeeling",
  "description": "Subjective user input regarding their physical and mental state on a specific day.",
  "type": "object",
  "properties": {
    "date": {
      "description": "The date this feeling log applies to",
      "type": "string",
      "format": "date-time"
    },
    "overall_feeling": {
      "description": "Rating scale, e.g., 1 (Very Poor) to 5 (Excellent)",
      "type": "integer",
      "minimum": 1,
      "maximum": 5
    },
    "energy_level": {
      "description": "Rating scale, e.g., 1 (Very Low) to 5 (Very High)",
      "type": "integer",
      "minimum": 1,
      "maximum": 5
    },
    "shin_pain": {
      "description": "Rating scale, e.g., 0 (None) to 5 (Severe)",
      "type": "integer",
      "minimum": 0,
      "maximum": 5
    },
    "sleep_quality": {
      "description": "Rating scale, e.g., 1 (Very Poor) to 5 (Excellent)",
      "type": "integer",
      "minimum": 1,
      "maximum": 5
    },
    "stress_level": {
      "description": "Optional: 1 (Very Low) to 5 (Very High)",
      "type": ["integer", "null"],
      "minimum": 1,
      "maximum": 5
    },
    "hydration_level": {
      "description": "1 (Poor) to 5 (Excellent). Note that this represents the previous day",
      "type": ["integer", "null"],
      "minimum": 1,
      "maximum": 5
    },
    "nutrition_quality": {
      "description": "1 (Poor) to 5 (Excellent). Note that this represents the previous day",
      "type": ["integer", "null"],
      "minimum": 1,
      "maximum": 5
    },
    "notes": {
      "description": "Optional free-text notes",
      "type": ["string", "null"]
    }
  },
  "required": [
    "date",
    "overall_feeling",
    "energy_level",
    "shin_pain",
    "sleep_quality"
  ]
}
"""

# --- 3. Workout Data ---
class WorkoutData(typing.TypedDict):
    """
    Data for a single workout session (running or other types).
    
    Captures detailed information about each completed workout including objective 
    measurements (duration, distance, pace) as well as subjective assessments
    (perceived exertion, pain levels). This data forms the foundation for tracking
    training load and progress over time.
    """
    date: datetime  # Date of the workout
    workout_type: str  # E.g., "Easy Run", "Tempo", "Strength", "Yoga", "Cycling", "Other"
    perceived_exertion: int  # RPE scale 1-10 (Rate of Perceived Exertion)
    shin_and_knee_pain: int  # 0 (None) to 5 (Severe) (Specific to this user's shin pain)
    shin_tightness: int  # 0 (None) to 5 (Severe) (Specific to this user's shin tightness)
    workout_adherence: int # Percent adherence to the planned workout (0-100%)
    actual_duration_minutes: float # Total workout time in minutes
    actual_distance_miles: typing.Optional[float]  # Distance in miles (None if not applicable)
    average_pace_minutes_per_mile: typing.Optional[float]  # Pace in min/mile (None if not applicable)
    average_heart_rate_bpm: typing.Optional[int]  # Average heart rate during workout if tracked
    max_heart_rate_bpm: typing.Optional[int]  # Maximum heart rate during workout if tracked
    elevation_gain_feet: typing.Optional[float]  # Elevation gain in feet (None if not applicable)
    notes: typing.Optional[str]  # User comments, could include exercises/sets/reps for strength

# JSON schema for WorkoutData
WorkoutDataSchema = """
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "WorkoutData",
  "description": "Data for a single workout session (running or other types).",
  "type": "object",
  "properties": {
    "date": {
      "description": "Date of the workout",
      "type": "string",
      "format": "date-time"
    },
    "workout_type": {
      "description": "E.g., \"Easy Run\", \"Tempo\", \"Strength\", \"Yoga\", \"Cycling\", \"Other\"",
      "type": "string"
    },
    "perceived_exertion": {
      "description": "RPE scale 1-10 (Made mandatory as it's often key)",
      "type": "integer",
      "minimum": 1,
      "maximum": 10
    },
    "shin_and_knee_pain": {
      "description": "0 (None) to 5 (Severe) (Specific to this user's shin pain)",
      "type": "integer",
      "minimum": 0,
      "maximum": 5
    },
    "shin_tightness": {
      "description": "0 (None) to 5 (Severe) (Specific to this user's shin tightness)",
      "type": "integer",
      "minimum": 0,
      "maximum": 5
    },
    "workout_adherence": {
      "description": "Percent adherence to the planned workout (0-100%)",
      "type": "integer",
      "minimum": 0,
      "maximum": 100
    },
    "actual_duration_minutes": {
      "description": "Total workout time in minutes",
      "type": "number",
       "exclusiveMinimum": 0
    },
    "actual_distance_miles": {
      "description": "Distance in miles (None if not applicable)",
      "type": ["number", "null"],
       "minimum": 0
    },
    "average_pace_minutes_per_mile": {
      "description": "Pace in min/mile (None if not applicable)",
      "type": ["number", "null"],
       "minimum": 0
    },
    "average_heart_rate_bpm": {
      "description": "Average Heart Rate (BPM) if tracked",
      "type": ["integer", "null"],
      "minimum": 0
    },
    "max_heart_rate_bpm": {
      "description": "Max Heart Rate (BPM) if tracked",
      "type": ["integer", "null"],
      "minimum": 0
    },
    "elevation_gain_feet": {
      "description": "Elevation gain in feet (None if not applicable)",
      "type": ["number", "null"]
    },
    "notes": {
      "description": "User comments, could include exercises/sets/reps for strength",
      "type": ["string", "null"]
    }
  },
  "required": [
    "date",
    "workout_type",
    "perceived_exertion",
    "shin_and_knee_pain",
    "shin_tightness",
    "workout_adherence",
    "actual_duration_minutes"
  ]
}
"""

# --- 4. Weekly User Summary ---
class WeeklyUserSummary(typing.TypedDict):
    """
    A qualitative summary of the user's training week.
    
    Provides both narrative and quantitative assessment of the user's weekly training,
    highlighting key achievements, areas for improvement, and overall progress.
    This aggregate view helps track broader patterns beyond individual workouts.
    """
    week_start_date: datetime  # E.g., the Monday of the week
    overall_summary: str  # Text summary of the week (consistency, feeling, progress)
    key_achievements: typing.Optional[typing.List[str]]  # Bullet points of successes
    areas_for_focus: typing.Optional[typing.List[str]]  # Bullet points for improvement areas
    total_workouts: int # Total number of workouts logged this week
    # Optional quantitative context if available/relevant
    total_running_distance_miles: typing.Optional[float]  # Total miles covered in running workouts
    total_workout_duration_minutes: typing.Optional[float]  # Total workout time across all activities

# JSON schema for WeeklyUserSummary
WeeklyUserSummarySchema = """
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "WeeklyUserSummary",
  "description": "A qualitative summary of the user's training week.",
  "type": "object",
  "properties": {
    "week_start_date": {
      "description": "E.g., the Monday of the week",
      "type": "string",
      "format": "date-time"
    },
    "overall_summary": {
      "description": "Text summary of the week (consistency, feeling, progress)",
      "type": "string"
    },
    "key_achievements": {
      "description": "Bullet points of successes",
      "type": ["array", "null"],
      "items": {
        "type": "string"
      }
    },
    "areas_for_focus": {
      "description": "Bullet points for improvement areas",
      "type": ["array", "null"],
      "items": {
        "type": "string"
      }
    },
    "total_workouts": {
      "description": "Total number of workouts logged this week",
      "type": "integer",
      "minimum": 0
    },
    "total_running_distance_miles": {
      "description": "Optional quantitative context if available/relevant",
      "type": ["number", "null"],
       "minimum": 0
    },
    "total_workout_duration_minutes": {
      "description": "Optional quantitative context if available/relevant",
      "type": ["number", "null"],
       "minimum": 0
    }
  },
  "required": [
    "week_start_date",
    "overall_summary",
    "total_workouts"
  ]
}
"""

def collect_weekly_summary(week_start_date: datetime, workouts: typing.List[WorkoutData], feelings: typing.List[DailyUserFeeling]) -> typing.Optional[WeeklyUserSummary]:
    """
    Collects or generates a weekly training summary.
    
    Aggregates existing workout and feeling data for a specific week and combines it with
    user input to create a qualitative assessment of the week's training.
    
    Args:
        week_start_date: The starting date of the week (typically Monday)
        workouts: List of workout data for filtering to the relevant week
        feelings: List of daily feelings for filtering to the relevant week
        
    Returns:
        A WeeklyUserSummary dictionary with the generated summary,
        or None if collection was skipped or failed
    """
    print(f"\n--- Logging Weekly Summary for week starting {week_start_date.date().isoformat()} ---")
    print("This will generate a summary of your training for the past week.")
    
    # Calculate the end date of the week (inclusive)
    week_end_date = week_start_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
    print(f"Analyzing data from {week_start_date.date().isoformat()} to {week_end_date.date().isoformat()}")
    
    # Filter workouts and feelings data to just this week
    print("Filtering workout and feeling data for the week...")
    week_workouts = [w for w in workouts if 
                    isinstance(w.get('date'), datetime) and 
                    week_start_date <= w.get('date') <= week_end_date]
    
    week_feelings = [f for f in feelings if 
                     isinstance(f.get('date'), datetime) and 
                     week_start_date <= f.get('date') <= week_end_date]
    
    print(f"Found {len(week_workouts)} workouts and {len(week_feelings)} feeling logs for the week")
    
    # Calculate aggregated metrics for the week
    print("Calculating weekly aggregated metrics...")
    total_wk_workouts = len(week_workouts)
    total_wk_duration = sum(w.get('actual_duration_minutes', 0) for w in week_workouts)
    
    # Sum running distances (only from workout types that are runs)
    total_wk_distance = sum(
        w.get('actual_distance_miles', 0) or 0  # Handle None values
        for w in week_workouts 
        if any(run_type in w.get('workout_type', '').lower() 
               for run_type in ['run', 'tempo', 'interval', 'sprint', 'jog'])
    )
    print(f"Weekly totals: {total_wk_workouts} workouts, {total_wk_duration:.1f} minutes, {total_wk_distance:.1f} running miles")

    # System prompt for the AI
    system_prompt = f"""
        You are an AI fitness coach assistant. Generate a concise weekly summary for the user's training week starting {week_start_date.date().isoformat()}.
        Base the summary on the provided workout logs and daily feelings for the week.
        Highlight consistency, overall feeling, progress, key achievements (e.g., longest run, PR), and areas needing focus (e.g., recurring pain, missed workouts).
        Calculate quantitative totals.
        The required output format is JSON matching this structure: {WeeklyUserSummarySchema}
        Set the 'week_start_date' field to {week_start_date.isoformat()}.
        Set 'total_workouts', 'total_running_distance_miles', and 'total_workout_duration_minutes' based on the calculations.
    """
    
    # User-facing prompt for additional context
    user_prompt = (
        f"Let's summarize your week starting {week_start_date.date().isoformat()}.\n"
        f"This week you logged {total_wk_workouts} workouts, totaling {total_wk_duration:.1f} minutes "
        f"and {total_wk_distance:.1f} running miles.\n"
        "Based on this, what's your overall summary? Any key achievements or areas to focus on next week? "
        "(The AI will also generate a summary based on the data)."
    )

    # Get user input for additional context
    print("Waiting for user to provide additional weekly context...")
    user_text, user_files = get_user_input_with_multimodal(user_prompt)
    print(f"Received user input: {len(user_text) if user_text else 0} characters of text and {len(user_files) if user_files else 0} files")

    # Combine AI context data and user input into a comprehensive context
    print("Preparing comprehensive context for weekly summary generation...")
    full_context = (
        f"Context for week starting {week_start_date.date().isoformat()}:\n"
        f"Total Workouts: {total_wk_workouts}\n"
        f"Total Duration (min): {total_wk_duration:.1f}\n"
        f"Total Running Distance (miles): {total_wk_distance:.1f}\n"
        f"Workout Details:\n{json.dumps(week_workouts, cls=DateEncoder, indent=2)}\n\n"
        f"Daily Feeling Details:\n{json.dumps(week_feelings, cls=DateEncoder, indent=2)}\n\n"
        "User's input/perspective:"
    )
    if user_text:
        full_context += "\n" + user_text

    # Use Gemini to generate the weekly summary
    print("Processing data to generate weekly summary...")
    extracted_data = call_gemini_for_structured_output(
        system_prompt=system_prompt,
        user_text=full_context, # Send combined context
        user_files=user_files,
        output_schema=WeeklyUserSummary,
    )

    # Process, validate and return the extracted data
    if extracted_data:
        print("Successfully generated weekly summary.")
        # Ensure required fields are set correctly regardless of AI output
        extracted_data['week_start_date'] = week_start_date
        extracted_data['total_workouts'] = total_wk_workouts
        
        # Only set distance and duration if there were values > 0
        extracted_data['total_running_distance_miles'] = total_wk_distance if total_wk_distance > 0 else None
        extracted_data['total_workout_duration_minutes'] = total_wk_duration if total_wk_duration > 0 else None

        # Add human-readable summary
        print("\n=== Weekly Summary ===")
        print(f"Week Starting: {week_start_date.date().isoformat()}")
        print(f"Total Workouts: {extracted_data.get('total_workouts')}")
        if extracted_data.get('total_running_distance_miles') is not None:
            print(f"Total Running Distance: {extracted_data.get('total_running_distance_miles'):.1f} miles")
        if extracted_data.get('total_workout_duration_minutes') is not None:
            print(f"Total Workout Duration: {extracted_data.get('total_workout_duration_minutes'):.1f} minutes")
        print("\nOverall Summary:")
        print(f"{extracted_data.get('overall_summary')}")
        if extracted_data.get('key_achievements'):
            print("\nKey Achievements:")
            for achievement in extracted_data.get('key_achievements'):
                print(f"- {achievement}")
        if extracted_data.get('areas_for_focus'):
            print("\nAreas for Focus:")
            for area in extracted_data.get('areas_for_focus'):
                print(f"- {area}")
        print("====================")
        input("Press Enter to continue...")
        
        return typing.cast(WeeklyUserSummary, extracted_data)
    else:
        print("Failed to generate weekly summary.")
        return None


# %%
def collect_workout_data(target_date: datetime) -> typing.Optional[WorkoutData]:
    """
    Prompts user for details about a completed workout.
    
    Collects comprehensive data about a workout including type, duration, distance,
    subjective assessments, and other metrics that help track training progress.
    
    Args:
        target_date: The date for which the workout data is being collected
        
    Returns:
        A WorkoutData dictionary with the collected information,
        or None if collection was skipped or failed
    """
    print(f"\n--- Logging Workout for {target_date.date().isoformat()} ---")
    print("This will capture details about your workout session.")
    
    # System prompt for the AI
    system_prompt = f"""
        You are an AI fitness coach assistant. Collect details about the user's workout completed on {target_date.date().isoformat()}.
        Infer workout_type if not explicitly stated (e.g., 'run', 'strength', 'yoga', 'cycle').
        Extract quantitative data like duration, distance, pace, HR, elevation where provided.
        Capture perceived exertion (RPE 1-10) and specific pain/tightness ratings (0-5).
        Determine adherence percentage (0-100%).
        The required output format is JSON matching this structure: {WorkoutDataSchema}
    """
    
    # User-facing prompt explaining what to provide
    user_prompt = (
        f"Did you complete a workout today ({target_date.date().isoformat()})? If yes, please describe it:\n"
        "- What type of workout was it (e.g., Easy Run, Tempo Run, Strength, Cycling, Yoga)?\n"
        "- How long did it last (minutes)?\n"
        "- If running/cycling: What distance (miles)? Average pace (min/mile)? Elevation gain (feet)?\n"
        "- How hard did it feel (Perceived Exertion, RPE 1-10)?\n"
        "- Any shin or knee pain (0=None to 5=Severe)?\n"
        "- Any shin tightness (0=None to 5=Severe)?\n"
        "- How well did you stick to the plan (Workout Adherence, 0-100%)?\n"
        "- Average/Max Heart Rate (BPM) if tracked?\n"
        "- Any notes (e.g., exercises done for strength, how the run felt)?\n"
        "You can also upload files like GPS data summaries (screenshots/text), photos, or videos of exercises."
    )
    
    # Get user input
    print("Waiting for user to input workout data...")
    user_text, user_files = get_user_input_with_multimodal(user_prompt)
    if not user_text and not user_files:
        print("User skipped workout log.")
        return None

    print(f"Received user input: {len(user_text) if user_text else 0} characters of text and {len(user_files) if user_files else 0} files")

    # Use Gemini to extract structured data
    print("Processing user input to extract workout data...")
    extracted_data = call_gemini_for_structured_output(
        system_prompt=system_prompt,
        user_text=user_text,
        user_files=user_files,
        output_schema=WorkoutData,
    )

    # Process, validate and return the extracted data
    if extracted_data:
        print("Successfully extracted workout data.")
        # Ensure the date field is set correctly
        extracted_data['date'] = target_date
        
        # Add human-readable summary
        print("\n=== Workout Summary ===")
        print(f"Date: {target_date.date().isoformat()}")
        print(f"Type: {extracted_data.get('workout_type')}")
        print(f"Duration: {extracted_data.get('actual_duration_minutes')} minutes")
        if extracted_data.get('actual_distance_miles') is not None:
            print(f"Distance: {extracted_data.get('actual_distance_miles')} miles")
        if extracted_data.get('average_pace_minutes_per_mile') is not None:
            print(f"Avg Pace: {extracted_data.get('average_pace_minutes_per_mile')} min/mile")
        print(f"Perceived Exertion: {extracted_data.get('perceived_exertion')}/10")
        print(f"Shin Pain: {extracted_data.get('shin_and_knee_pain')}/5")
        print(f"Shin Tightness: {extracted_data.get('shin_tightness')}/5")
        print(f"Workout Adherence: {extracted_data.get('workout_adherence')}%")
        if extracted_data.get('average_heart_rate_bpm') is not None:
            print(f"Avg HR: {extracted_data.get('average_heart_rate_bpm')} bpm")
        if extracted_data.get('max_heart_rate_bpm') is not None:
            print(f"Max HR: {extracted_data.get('max_heart_rate_bpm')} bpm")
        if extracted_data.get('elevation_gain_feet') is not None:
            print(f"Elevation Gain: {extracted_data.get('elevation_gain_feet')} feet")
        if extracted_data.get('notes'):
            print(f"Notes: {extracted_data.get('notes')}")
        print("=====================")
        input("Press Enter to continue...")
        
        return typing.cast(WorkoutData, extracted_data)
    else:
        print("Failed to extract workout data from input.")
        return None


# %% [markdown]
# ### Planning Mode

# %% [markdown]
# Okay, now the planning phase. This is where our work starts to pay off. We've gathered ALL that data, and we use the huge context window to upload it ALL to the LLM to help us plan. We also have a detailed system prompt with an informal format for the LLM to follow, which we render in markdown. The user can provide feedback to rework the plan based on their daily needs.

# %%
def plan_workout(all_data: typing.Dict[str, typing.Any]) -> typing.Optional[str]:
    """
    Generates a personalized workout plan based on all available user data.
    
    Uses the Gemini API to analyze all user data (profile, feelings, workouts,
    summaries, stats) and create a structured workout plan tailored to the user's
    goals, preferences, and current fitness level. Supports an interactive
    refinement process where the user can provide feedback on the plan.
    
    Args:
        all_data: Dictionary containing all user data categories (profile, workouts, etc.)
        
    Returns:
        The accepted workout plan as text (can be markdown formatted),
        or None if no plan was accepted or generated
    """
    print("\n--- Generating Workout Plan ---")
    print("Analyzing your data to create a personalized training plan...")

    # Convert all data to JSON for inclusion in the prompt
    print("Preparing user data context for AI...")
    context_json = json.dumps(all_data, cls=DateEncoder, indent=2)

    # Check if the context data is too large and warn if needed
    if len(context_json) > 250000: # Example limit, adjust based on model/API limits
        print("Warning: Context data is very large, may need pruning...")
        print(f"Context size: {len(context_json)} characters")

    # System prompt instructing the AI how to create the workout plan
    system_prompt = """
        You are an expert AI running coach. Analyze the provided user data (static info, goals,
        daily feelings, workout logs, weekly summaries, monthly stats) to generate the
        next logical workout plan for the user. Think thoroughly, step by step through the 
        user's data and goals.

        Consider:
        - User's goal, goal date, and current fitness level (inferred from data).
        - Recent workout performance, perceived exertion, and adherence.
        - Subjective feedback: energy levels, sleep, stress, pain (especially shin issues).
        - Weekly/monthly trends and summaries.
        - User's preferred training days and long run day.
        - General principles of training periodization (base building, intensity, tapering).

        Response Format:
        1. Workout Feedback: 
            • Concise evaluation of the previous workout.
        2. Next Week's Plan:
            • A short table listing workouts for the coming week.
        3. Missing Data/Inputs:
            • List any additional data or clarifications you need from me for next time. 
        4. Detailed Next Workout: 
            • Provide details on the next scheduled workout and a fallback option, and why this 
            workout was chosen.
        5. Goal Progress Status:
            • Update on how many weeks remain until the user can realistically complete their stated goal, why, and what they can do to improve.
        6. Summary & Motivation:
            • A brief summary of my performance, focus areas, and motivational insights.
            • Include a motivational or funny quote.

        Be prepared to revise the plan based on user feedback. Output the plan as markdown.
    """

    # Interactive planning loop - continues until plan is accepted or user exits
    iteration = 0
    while True:
        iteration += 1
        print(f"Generating workout plan (iteration {iteration})...")
        try:
            # Call Gemini to generate the workout plan
            print("Sending request to Gemini to generate workout plan...")
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[system_prompt, f"User Data:\n{context_json}"],
                config=genai.types.GenerateContentConfig(
                    temperature=0.5, # Allow some creativity in planning
                ),
            )
            print("Received response from Gemini")

            # Handle the response
            if response.candidates and response.candidates[0].content.parts:
                 # Extract the plan text
                 plan_text = response.candidates[0].content.parts[0].text
                 
                 # Display the plan to the user with rich formatting
                 print("\n--- Proposed Workout Plan ---")
                 from rich.console import Console
                 from rich.markdown import Markdown
                 console = Console()
                 console.print(Markdown(plan_text))
                 print("-----------------------------")

                 # Get user feedback on the plan
                 print("Waiting for user feedback on the workout plan...")
                 feedback = input("What do you think? Enter feedback, ask questions, or type 'accept': ")

                 # Process user response
                 if feedback.lower() == 'accept':
                     print("Workout plan accepted! Saving plan...")
                     return plan_text
                 elif feedback.lower() in ['quit', 'exit']:
                     print("Exiting planning phase without accepting a plan.")
                     raise Exception("Exiting planning phase without a plan.")
                 else:
                     # Include the user feedback for plan refinement
                     print(f"User provided feedback: {len(feedback)} characters")
                     print("Preparing to revise workout plan based on feedback...")
                     context_json = json.dumps({
                         "user_data": all_data, 
                         "previous_plan": plan_text, 
                         "user_feedback": feedback
                     }, cls=DateEncoder)

                     print("Asking AI to revise plan based on feedback...")
                     # Loop continues for another iteration

            else:
                 print("Error: No valid content in Gemini response")
                 raise Exception("Failed to generate a response during workout planning.")

        except Exception as e:
            print(f"Error during workout planning: {e}")
            raise Exception(f"Error during workout planning: {e}")


# %% [markdown]
# ### Q&A Mode

# %% [markdown]
# Last but not least, a Q&A mode to ask qeustions and get coaching advice without modifying the plan itself. Instead, the plan is part of the input to the LLM. 

# %%
def workout_q_and_a(all_data: typing.Dict[str, typing.Any], workout_plan: str) -> None:
    """
    Handles user questions about their workout plan and fitness data.
    
    Creates an interactive Q&A session where users can ask questions about their
    plan, get clarification on training concepts, or request fitness advice
    specific to their situation. Uses the Gemini chat API to maintain conversation
    context across multiple questions.
    
    Args:
        all_data: Dictionary containing all user data categories
        workout_plan: The accepted workout plan text from plan_workout()
    """
    print("\n--- Workout Q&A ---")
    print("Starting interactive Q&A session. Ask any questions about your plan, data, or general fitness advice.")

    # Prepare context data for the AI
    print("Preparing context data for Q&A session...")
    context_json = json.dumps({"user_data": all_data, "accepted_plan": workout_plan}, cls=DateEncoder, indent=2)
    
    # Check if the context is too large
    if len(context_json) > 250000:
        print("Warning: Q&A context data is very large, may need pruning...")
        print(f"Context size: {len(context_json)} characters")

    # System prompt for the AI
    system_prompt = """
        You are an AI fitness coach assistant. The user has accepted a workout plan.
        Your role now is to answer the user's questions clearly and concisely.
        Refer to the user's complete data history (profile, goals, logs, summaries, stats)
        and the accepted workout plan provided below.
        Explain the reasoning behind the plan, define terms, offer encouragement,
        or provide general fitness advice based on their context.
        Be helpful and supportive.
    """

    # Use the Chat API for conversation history (better for Q&A)
    print("Initializing chat session with Gemini...")
    chat_client = genai.GenerativeModel(model_name='gemini-2.0-flash', system_instruction=system_prompt)
    
    # Initialize chat with context data
    print("Loading user data and workout plan into chat context...")
    chat = chat_client.start_chat(history=[
        {'role': 'user', 'parts': [f"Here is my data and the workout plan we agreed on:\n{context_json}"]},
        {'role': 'model', 'parts': ["Okay, I have reviewed your data and the workout plan. How can I help you?"]}
    ])
    print("Chat session initialized successfully")

    # Interactive Q&A loop
    question_count = 0
    while True:
        # Get user question
        user_question = input("\nYour question (or type 'done'): ")
        
        # Check for exit command
        if user_question.lower() == 'done':
            print("Exiting Q&A session.")
            break
        
        # Skip empty questions
        if not user_question:
            print("Empty question, please try again.")
            continue

        question_count += 1
        try:
            # Send the question to Gemini
            print(f"Processing question #{question_count}: {user_question[:30]}...")
            response = chat.send_message(user_question, stream=False)
            print("Received response from Gemini")

            # Display the response
            if response.candidates and response.candidates[0].content.parts:
                 answer = response.candidates[0].content.parts[0].text
                 print(f"\nAI Coach: {answer}")
            else:
                 print("AI Coach: Sorry, I couldn't generate a response for that.")

        except Exception as e:
            print(f"Error during Q&A: {e}")
            raise Exception(f"Error during Q&A: {e}")

# %% [markdown]
# ## Putting it all together
# 

# %% [markdown]
# Now we're on to the actual execution of the program. It's pretty straight forward, actually. Gather the data, then process the plan. We collect the data from most granular to least granular, because we can use some of the very granular parts to automatically aggregate statistics that feed into the less granular parts, like the total running distance in a month.

# %%
"""
Main function to run the AI fitness coach CLI application.

This function orchestrates the overall flow of the application:
1. Loads existing user data or collects new data if it doesn't exist
2. Checks for missing data entries and prompts the user to fill them in
3. Generates a personalized workout plan based on all available data
4. Provides an interactive Q&A session for the user to ask questions

The application maintains persistent data across sessions by saving all
user information to JSON files.
"""
print("\n===========================================")
print("--- AI Fitness Coach Prototype Starting ---")
print("===========================================")

print("\nStarting application initialization...")

# --- 1. Load or Initialize Data ---
# Load all existing data from files, or initialize empty collections if files don't exist
print("\nLoading existing user data files...")
user_data: typing.Optional[StaticUserData] = load_json_data(USER_DATA_FILE)
daily_feelings: list[DailyUserFeeling] = load_json_data(DAILY_FEELING_FILE) or []
workout_data: list[WorkoutData] = load_json_data(WORKOUT_DATA_FILE) or []
weekly_summaries: list[WeeklyUserSummary] = load_json_data(WEEKLY_SUMMARY_FILE) or []
monthly_stats: list[MonthlyUserStats] = load_json_data(MONTHLY_STATS_FILE) or []

print(f"\nData loaded: {len(daily_feelings)} feeling logs, {len(workout_data)} workouts, {len(weekly_summaries)} weekly summaries, {len(monthly_stats)} monthly stats")

# If no user profile exists, collect it (required for the app to function)
if user_data is None:
    print("No user profile found. Need to create one before proceeding...")
    user_data = collect_static_user_data()
    if user_data:
        print("Saving new user profile...")
        save_json_data(USER_DATA_FILE, user_data)
    else:
        raise("User data collection failed or skipped. Exiting.")

# Create empty data files if they don't exist yet
initialize_data_files()

# --- 2. Check for Missing Data & Prompt User ---
print("\nChecking for missing or outdated data...")
# Get today's date at midnight for consistent comparisons
today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
print(f"Current date: {today.date().isoformat()}")

# Get the most recent daily feeling date if any exist
last_daily_log_date = None
if daily_feelings and 'date' in daily_feelings[-1]:
    last_daily_log_date = normalize_date_to_datetime(daily_feelings[-1]['date'])
    print(f"Last daily feeling log: {last_daily_log_date.date().isoformat() if last_daily_log_date else 'None'}")

# Get the most recent workout date if any exist
last_workout_log_date = None
if workout_data and 'date' in workout_data[-1]:
    last_workout_log_date = normalize_date_to_datetime(workout_data[-1]['date'])
    print(f"Last workout log: {last_workout_log_date.date().isoformat() if last_workout_log_date else 'None'}")
    

# %%
# --- Check for Missing Workout Log ---
# Print summary of last recorded workout, if present:
if last_workout_log_date is not None:
    print(f"Last workout: {last_workout_log_date.date().isoformat()}")
    print(f"Workout type: {workout_data[-1]['workout_type']}")
    print(f"Duration: {workout_data[-1]['actual_duration_minutes']} minutes")
    print(f"Distance: {workout_data[-1]['actual_distance_miles']} miles")
    print(f"Pace: {workout_data[-1]['average_pace_minutes_per_mile']} minutes per mile")
    print(f"Heart rate: {workout_data[-1]['average_heart_rate_bpm']} bpm")
    print(f"Elevation gain: {workout_data[-1]['elevation_gain_feet']} feet")
          

# Only prompt for workout if feeling is logged for today but workout isn't
while True:
     # Ask if user actually did a workout before prompting for details
     ask_workout = input(f"Do you have any more workouts to log since {last_workout_log_date.date().isoformat() if last_workout_log_date else 'your last entry'}? (yes/no): ")
     if ask_workout.lower() in ['yes', 'y']:
         new_workout = collect_workout_data(today)
         if new_workout:
             workout_data.append(new_workout)
             save_json_data(WORKOUT_DATA_FILE, workout_data)
     else:
          print("Okay, no workout logged for today.")
          break

print("\n--- All data checks complete ---")

# %%
# --- Check for Missing Daily Feeling Log ---
if last_daily_log_date is None or last_daily_log_date.date() < today.date():
    print(f"\nLooks like you haven't logged your feeling for today ({today.date().isoformat()}).")
    new_feeling = collect_daily_feeling(today)
    if new_feeling:
        daily_feelings.append(new_feeling)
        save_json_data(DAILY_FEELING_FILE, daily_feelings)
else:
     print(f"\nDaily feeling log is up-to-date ({last_daily_log_date.date().isoformat()}).")

# %%
# --- Check for Missing Weekly Summary ---
# Find the Monday of the current week
today_date = today.date()
last_monday_date = today_date - timedelta(days=today_date.weekday())
last_monday = datetime.combine(last_monday_date, datetime.min.time())

# Get the most recent weekly summary date if any exist
last_summary_date = None
if weekly_summaries and 'week_start_date' in weekly_summaries[-1]:
    last_summary_date = normalize_date_to_datetime(weekly_summaries[-1]['week_start_date'])

# Check if we need a summary for the previous week
if last_summary_date is None or last_summary_date < last_monday:
     # Calculate the start of the previous week
     prev_week_start_date = last_monday_date - timedelta(days=7)
     prev_week_start = datetime.combine(prev_week_start_date, datetime.min.time())
     
     # Only collect summary if it doesn't already exist for the previous week
     if last_summary_date is None or last_summary_date < prev_week_start:
        print(f"\nLooks like the summary for the week starting {prev_week_start.date().isoformat()} is missing.")
        new_summary = collect_weekly_summary(prev_week_start, workout_data, daily_feelings)
        if new_summary:
            weekly_summaries.append(new_summary)
            save_json_data(WEEKLY_SUMMARY_FILE, weekly_summaries)
else:
     print(f"\nWeekly summary is up-to-date (last summary for week starting {last_summary_date.date().isoformat()}).")

# %%
# --- Check for Missing Monthly Stats ---
print("\nChecking for missing monthly statistics...")
current_month_str = f"{today.year}-{today.month:02d}"
print(f"Current month: {current_month_str}")

# Get the most recent monthly stats if any exist
last_month_stat = None
if monthly_stats and 'month' in monthly_stats[-1]:
    last_month_stat = monthly_stats[-1]['month']
    print(f"Last monthly stats: {last_month_stat}")

# Check if we need to generate stats for previous month
if last_month_stat is None or last_month_stat < current_month_str:
     # Calculate the previous month
     prev_month_date = today.replace(day=1) - timedelta(days=1)
     prev_month_str = f"{prev_month_date.year}-{prev_month_date.month:02d}"
     print(f"Previous month: {prev_month_str}")

     # Only collect stats if they don't already exist for the previous month
     if last_month_stat is None or last_month_stat < prev_month_str:
         print(f"Monthly stats for {prev_month_str} are missing. Prompting user for data...")
         new_stats = collect_monthly_stats(prev_month_date.year, prev_month_date.month, user_data, workout_data)
         if new_stats:
             print(f"Saving new monthly stats for {prev_month_str}...")
             monthly_stats.append(new_stats)
             save_json_data(MONTHLY_STATS_FILE, monthly_stats)
else:
     print(f"\nMonthly stats are up-to-date (last stats for {last_month_stat}).")

# %%
# --- 3. Workout Planning Phase ---
# Prepare all data for workout planning
all_current_data = {
    "user_profile": user_data,
    "daily_feelings": daily_feelings,
    "workout_logs": workout_data,
    "weekly_summaries": weekly_summaries,
    "monthly_stats": monthly_stats
}

accepted_plan = plan_workout(all_current_data)

# %%
# 4. Q&A Phase
if accepted_plan:
    workout_q_and_a(all_current_data, accepted_plan)
else:
    print("\nNo workout plan was accepted or generated.")

print("\n--- AI Fitness Coach session finished ---")

# %% [markdown]
# This is a helper function to pre-load the workout_data.json file, which makes the whole thing more interesting. That said, it is optional, which is why it's at the end.

# %%
workouts = """[
  {
    "date": "2025-03-01T07:10:49",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 58.61516666666667,
    "actual_distance_miles": 6.36692144747,
    "average_pace_minutes_per_mile": 9.206202267495911,
    "average_heart_rate_bpm": 163,
    "max_heart_rate_bpm": 180,
    "elevation_gain_feet": 121.39108,
    "notes": null
  },
  {
    "date": "2025-03-02T17:04:17",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 67.91516666666666,
    "actual_distance_miles": 4.03200185448,
    "average_pace_minutes_per_mile": 16.844031604599934,
    "average_heart_rate_bpm": 102,
    "max_heart_rate_bpm": 126,
    "elevation_gain_feet": 91.86352,
    "notes": null
  },
  {
    "date": "2025-03-02T19:38:13",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 48.26933333333333,
    "actual_distance_miles": 4.75269900883,
    "average_pace_minutes_per_mile": 10.156194037041718,
    "average_heart_rate_bpm": 147,
    "max_heart_rate_bpm": 169,
    "elevation_gain_feet": 78.74016,
    "notes": null
  },
  {
    "date": "2025-03-03T06:54:46",
    "workout_type": "Indoor Sport & Fitness",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 25.45583333333333,
    "actual_distance_miles": 0.0,
    "average_pace_minutes_per_mile": null,
    "average_heart_rate_bpm": 85,
    "max_heart_rate_bpm": 122,
    "elevation_gain_feet": 0.0,
    "notes": null
  },
  {
    "date": "2025-03-03T20:05:48",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 41.746833333333335,
    "actual_distance_miles": 4.444287726690001,
    "average_pace_minutes_per_mile": 9.393368724221952,
    "average_heart_rate_bpm": 154,
    "max_heart_rate_bpm": 177,
    "elevation_gain_feet": 72.17848,
    "notes": null
  },
  {
    "date": "2025-03-04T06:04:31",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 35.095,
    "actual_distance_miles": 3.1497358127100004,
    "average_pace_minutes_per_mile": 11.142204326592275,
    "average_heart_rate_bpm": 136,
    "max_heart_rate_bpm": 156,
    "elevation_gain_feet": 55.77428,
    "notes": null
  },
  {
    "date": "2025-03-04T06:39:45",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 20.125666666666667,
    "actual_distance_miles": 1.1627218426200001,
    "average_pace_minutes_per_mile": 17.30909829759182,
    "average_heart_rate_bpm": 102,
    "max_heart_rate_bpm": 169,
    "elevation_gain_feet": 16.4042,
    "notes": null
  },
  {
    "date": "2025-03-04T07:41:43",
    "workout_type": "Indoor Sport & Fitness",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 22.37316666666667,
    "actual_distance_miles": 0.0,
    "average_pace_minutes_per_mile": null,
    "average_heart_rate_bpm": 79,
    "max_heart_rate_bpm": 117,
    "elevation_gain_feet": 0.0,
    "notes": null
  },
  {
    "date": "2025-03-05T06:40:49",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 44.098499999999994,
    "actual_distance_miles": 4.70118113922,
    "average_pace_minutes_per_mile": 9.380302246196075,
    "average_heart_rate_bpm": 159,
    "max_heart_rate_bpm": 182,
    "elevation_gain_feet": 62.33596,
    "notes": null
  },
  {
    "date": "2025-03-05T08:15:36",
    "workout_type": "Indoor Sport & Fitness",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 48.96783333333334,
    "actual_distance_miles": 0.0,
    "average_pace_minutes_per_mile": null,
    "average_heart_rate_bpm": 85,
    "max_heart_rate_bpm": 126,
    "elevation_gain_feet": 0.0,
    "notes": null
  },
  {
    "date": "2025-03-06T16:44:29",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 91.52749999999999,
    "actual_distance_miles": 5.4664492354,
    "average_pace_minutes_per_mile": 16.74350132207943,
    "average_heart_rate_bpm": 93,
    "max_heart_rate_bpm": 116,
    "elevation_gain_feet": 118.11024,
    "notes": null
  },
  {
    "date": "2025-03-07T17:48:08",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 96.6315,
    "actual_distance_miles": 9.69150484587,
    "average_pace_minutes_per_mile": 9.970742576802113,
    "average_heart_rate_bpm": 150,
    "max_heart_rate_bpm": 169,
    "elevation_gain_feet": 160.76116,
    "notes": null
  },
  {
    "date": "2025-03-08T16:11:04",
    "workout_type": "Hiking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 73.729,
    "actual_distance_miles": 2.58861915858,
    "average_pace_minutes_per_mile": 28.481980346790145,
    "average_heart_rate_bpm": 96,
    "max_heart_rate_bpm": 134,
    "elevation_gain_feet": 380.57744,
    "notes": null
  },
  {
    "date": "2025-03-08T21:16:36",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 35.22866666666666,
    "actual_distance_miles": 2.15548628932,
    "average_pace_minutes_per_mile": 16.343721062489518,
    "average_heart_rate_bpm": 101,
    "max_heart_rate_bpm": 115,
    "elevation_gain_feet": 49.2126,
    "notes": null
  },
  {
    "date": "2025-03-09T10:35:14",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 51.13433333333333,
    "actual_distance_miles": 5.27331470118,
    "average_pace_minutes_per_mile": 9.696810494145378,
    "average_heart_rate_bpm": 158,
    "max_heart_rate_bpm": 181,
    "elevation_gain_feet": 95.14436,
    "notes": null
  },
  {
    "date": "2025-03-10T06:29:34",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 16.051333333333336,
    "actual_distance_miles": 1.55023365306,
    "average_pace_minutes_per_mile": 10.354138101472428,
    "average_heart_rate_bpm": 134,
    "max_heart_rate_bpm": 147,
    "elevation_gain_feet": 52.49344,
    "notes": null
  },
  {
    "date": "2025-03-10T06:45:41",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 24.310166666666664,
    "actual_distance_miles": 1.47465629833,
    "average_pace_minutes_per_mile": 16.485310301930785,
    "average_heart_rate_bpm": 94,
    "max_heart_rate_bpm": 146,
    "elevation_gain_feet": 13.12336,
    "notes": null
  },
  {
    "date": "2025-03-10T08:04:47",
    "workout_type": "Indoor Sport & Fitness",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 20.799666666666667,
    "actual_distance_miles": 0.0,
    "average_pace_minutes_per_mile": null,
    "average_heart_rate_bpm": 76,
    "max_heart_rate_bpm": 111,
    "elevation_gain_feet": 0.0,
    "notes": null
  },
  {
    "date": "2025-03-11T13:18:44",
    "workout_type": "Indoor Sport & Fitness",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 24.807666666666666,
    "actual_distance_miles": 0.0,
    "average_pace_minutes_per_mile": null,
    "average_heart_rate_bpm": 81,
    "max_heart_rate_bpm": 117,
    "elevation_gain_feet": 0.0,
    "notes": null
  },
  {
    "date": "2025-03-11T16:25:08",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 7.955333333333333,
    "actual_distance_miles": 0.7739921450199999,
    "average_pace_minutes_per_mile": 10.278312751000552,
    "average_heart_rate_bpm": 146,
    "max_heart_rate_bpm": 162,
    "elevation_gain_feet": 26.24672,
    "notes": null
  },
  {
    "date": "2025-03-11T16:33:08",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 16.74233333333333,
    "actual_distance_miles": 0.9852458575999999,
    "average_pace_minutes_per_mile": 16.993051231006092,
    "average_heart_rate_bpm": 108,
    "max_heart_rate_bpm": 127,
    "elevation_gain_feet": 22.96588,
    "notes": null
  },
  {
    "date": "2025-03-11T17:02:16",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 58.198166666666665,
    "actual_distance_miles": 3.3447220325100004,
    "average_pace_minutes_per_mile": 17.400000986925857,
    "average_heart_rate_bpm": 100,
    "max_heart_rate_bpm": 109,
    "elevation_gain_feet": 62.33596,
    "notes": null
  },
  {
    "date": "2025-03-12T06:20:36",
    "workout_type": "Indoor Sport & Fitness",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 24.801166666666667,
    "actual_distance_miles": 0.0,
    "average_pace_minutes_per_mile": null,
    "average_heart_rate_bpm": 85,
    "max_heart_rate_bpm": 123,
    "elevation_gain_feet": 0.0,
    "notes": null
  },
  {
    "date": "2025-03-12T20:03:16",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 4.386166666666667,
    "actual_distance_miles": 0.42131439284,
    "average_pace_minutes_per_mile": 10.410673694530951,
    "average_heart_rate_bpm": 133,
    "max_heart_rate_bpm": 147,
    "elevation_gain_feet": 0.0,
    "notes": null
  },
  {
    "date": "2025-03-12T20:07:41",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 42.93333333333333,
    "actual_distance_miles": 2.64885486332,
    "average_pace_minutes_per_mile": 16.20826189001609,
    "average_heart_rate_bpm": 102,
    "max_heart_rate_bpm": 148,
    "elevation_gain_feet": 45.93176,
    "notes": null
  },
  {
    "date": "2025-03-13T16:56:21",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 71.36933333333333,
    "actual_distance_miles": 4.28455188372,
    "average_pace_minutes_per_mile": 16.657362373067578,
    "average_heart_rate_bpm": 99,
    "max_heart_rate_bpm": 111,
    "elevation_gain_feet": 19.68504,
    "notes": null
  },
  {
    "date": "2025-03-14T22:44:25",
    "workout_type": "Stair Climbing",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 43.07083333333333,
    "actual_distance_miles": 0.0,
    "average_pace_minutes_per_mile": null,
    "average_heart_rate_bpm": 119,
    "max_heart_rate_bpm": 130,
    "elevation_gain_feet": 0.0,
    "notes": null
  },
  {
    "date": "2025-03-15T17:00:12",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 93.22633333333333,
    "actual_distance_miles": 5.63339298197,
    "average_pace_minutes_per_mile": 16.548878026388287,
    "average_heart_rate_bpm": 102,
    "max_heart_rate_bpm": 126,
    "elevation_gain_feet": 154.19948,
    "notes": null
  },
  {
    "date": "2025-03-16T20:54:14",
    "workout_type": "Indoor Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 104.24633333333333,
    "actual_distance_miles": 5.75248494783,
    "average_pace_minutes_per_mile": 18.121965425160823,
    "average_heart_rate_bpm": 96,
    "max_heart_rate_bpm": 105,
    "elevation_gain_feet": 0.0,
    "notes": null
  },
  {
    "date": "2025-03-17T17:01:35",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 103.67366666666666,
    "actual_distance_miles": 6.531149802770001,
    "average_pace_minutes_per_mile": 15.873723585806658,
    "average_heart_rate_bpm": 105,
    "max_heart_rate_bpm": 122,
    "elevation_gain_feet": 144.35696,
    "notes": null
  },
  {
    "date": "2025-03-18T20:29:33",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 35.46383333333333,
    "actual_distance_miles": 3.3339598867899998,
    "average_pace_minutes_per_mile": 10.637150576961076,
    "average_heart_rate_bpm": 141,
    "max_heart_rate_bpm": 152,
    "elevation_gain_feet": 49.2126,
    "notes": null
  },
  {
    "date": "2025-03-19T16:56:36",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 93.20216666666667,
    "actual_distance_miles": 5.527902827299999,
    "average_pace_minutes_per_mile": 16.860312052950743,
    "average_heart_rate_bpm": 100,
    "max_heart_rate_bpm": 124,
    "elevation_gain_feet": 137.79528,
    "notes": null
  },
  {
    "date": "2025-03-20T19:05:09",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 37.63533333333333,
    "actual_distance_miles": 3.69075111499,
    "average_pace_minutes_per_mile": 10.197201642906041,
    "average_heart_rate_bpm": 144,
    "max_heart_rate_bpm": 156,
    "elevation_gain_feet": 72.17848,
    "notes": null
  },
  {
    "date": "2025-03-21T17:01:12",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 86.7155,
    "actual_distance_miles": 5.27378694314,
    "average_pace_minutes_per_mile": 16.442738573805524,
    "average_heart_rate_bpm": 98,
    "max_heart_rate_bpm": 122,
    "elevation_gain_feet": 147.6378,
    "notes": null
  },
  {
    "date": "2025-03-22T14:59:14",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 46.31066666666666,
    "actual_distance_miles": 4.0456223068,
    "average_pace_minutes_per_mile": 11.44710582320706,
    "average_heart_rate_bpm": 143,
    "max_heart_rate_bpm": 152,
    "elevation_gain_feet": 59.05512,
    "notes": null
  },
  {
    "date": "2025-03-23T09:03:43",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 74.19916666666667,
    "actual_distance_miles": 6.99538229058,
    "average_pace_minutes_per_mile": 10.606878021031596,
    "average_heart_rate_bpm": 148,
    "max_heart_rate_bpm": 171,
    "elevation_gain_feet": 137.79528,
    "notes": null
  },
  {
    "date": "2025-03-24T18:21:03",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 40.49083333333333,
    "actual_distance_miles": 3.6935721393299996,
    "average_pace_minutes_per_mile": 10.962513200210086,
    "average_heart_rate_bpm": 136,
    "max_heart_rate_bpm": 146,
    "elevation_gain_feet": 45.93176,
    "notes": null
  },
  {
    "date": "2025-03-25T17:00:15",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 68.87966666666667,
    "actual_distance_miles": 4.22252041679,
    "average_pace_minutes_per_mile": 16.312453195674454,
    "average_heart_rate_bpm": 102,
    "max_heart_rate_bpm": 115,
    "elevation_gain_feet": 32.8084,
    "notes": null
  },
  {
    "date": "2025-03-26T11:29:30",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 57.97083333333333,
    "actual_distance_miles": 5.27125796317,
    "average_pace_minutes_per_mile": 10.997532987073004,
    "average_heart_rate_bpm": 156,
    "max_heart_rate_bpm": 171,
    "elevation_gain_feet": 91.86352,
    "notes": null
  },
  {
    "date": "2025-03-27T17:06:30",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 63.527,
    "actual_distance_miles": 3.8554082162800003,
    "average_pace_minutes_per_mile": 16.47737319533334,
    "average_heart_rate_bpm": 99,
    "max_heart_rate_bpm": 126,
    "elevation_gain_feet": 82.021,
    "notes": null
  },
  {
    "date": "2025-03-28T22:55:31",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 18.169666666666668,
    "actual_distance_miles": 1.6945408541,
    "average_pace_minutes_per_mile": 10.72247188535144,
    "average_heart_rate_bpm": 134,
    "max_heart_rate_bpm": 146,
    "elevation_gain_feet": 36.08924,
    "notes": null
  },
  {
    "date": "2025-03-28T23:13:54",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 18.624166666666667,
    "actual_distance_miles": 1.09737846826,
    "average_pace_minutes_per_mile": 16.971507283350554,
    "average_heart_rate_bpm": 103,
    "max_heart_rate_bpm": 142,
    "elevation_gain_feet": 16.4042,
    "notes": null
  },
  {
    "date": "2025-03-29T17:12:07",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 107.096,
    "actual_distance_miles": 6.489685715939999,
    "average_pace_minutes_per_mile": 16.502494063302674,
    "average_heart_rate_bpm": 101,
    "max_heart_rate_bpm": 135,
    "elevation_gain_feet": 127.95276,
    "notes": null
  },
  {
    "date": "2025-03-30T20:23:20",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 15.115666666666668,
    "actual_distance_miles": 1.27009475142,
    "average_pace_minutes_per_mile": 11.901211818855991,
    "average_heart_rate_bpm": 139,
    "max_heart_rate_bpm": 164,
    "elevation_gain_feet": 39.37008,
    "notes": null
  },
  {
    "date": "2025-03-30T20:38:28",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 15.039666666666667,
    "actual_distance_miles": 0.90192622021,
    "average_pace_minutes_per_mile": 16.675052049340472,
    "average_heart_rate_bpm": 112,
    "max_heart_rate_bpm": 126,
    "elevation_gain_feet": 13.12336,
    "notes": null
  },
  {
    "date": "2025-03-30T22:43:25",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 21.063499999999998,
    "actual_distance_miles": 1.9729709854900002,
    "average_pace_minutes_per_mile": 10.676031302492135,
    "average_heart_rate_bpm": 141,
    "max_heart_rate_bpm": 156,
    "elevation_gain_feet": 45.93176,
    "notes": null
  },
  {
    "date": "2025-03-31T16:52:08",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 71.562,
    "actual_distance_miles": 4.3063185098500005,
    "average_pace_minutes_per_mile": 16.61790688178629,
    "average_heart_rate_bpm": 97,
    "max_heart_rate_bpm": 128,
    "elevation_gain_feet": 82.021,
    "notes": null
  },
  {
    "date": "2025-04-01T18:23:58",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 32.1655,
    "actual_distance_miles": 3.34016738308,
    "average_pace_minutes_per_mile": 9.62990662172741,
    "average_heart_rate_bpm": 154,
    "max_heart_rate_bpm": 168,
    "elevation_gain_feet": 49.2126,
    "notes": null
  },
  {
    "date": "2025-04-02T06:25:03",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 50.28933333333334,
    "actual_distance_miles": 5.2718731204600005,
    "average_pace_minutes_per_mile": 9.539177477197196,
    "average_heart_rate_bpm": 160,
    "max_heart_rate_bpm": 173,
    "elevation_gain_feet": 88.58268,
    "notes": null
  },
  {
    "date": "2025-04-03T16:52:43",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 87.88733333333333,
    "actual_distance_miles": 5.34786679376,
    "average_pace_minutes_per_mile": 16.434091708470014,
    "average_heart_rate_bpm": 101,
    "max_heart_rate_bpm": 113,
    "elevation_gain_feet": 19.68504,
    "notes": null
  },
  {
    "date": "2025-04-04T18:02:03",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 28.310166666666664,
    "actual_distance_miles": 3.1463120585,
    "average_pace_minutes_per_mile": 8.997889001564422,
    "average_heart_rate_bpm": 169,
    "max_heart_rate_bpm": 191,
    "elevation_gain_feet": 65.6168,
    "notes": null
  },
  {
    "date": "2025-04-05T17:03:25",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 93.63966666666667,
    "actual_distance_miles": 5.86791082479,
    "average_pace_minutes_per_mile": 15.95792258312273,
    "average_heart_rate_bpm": 102,
    "max_heart_rate_bpm": 130,
    "elevation_gain_feet": 114.82939999999999,
    "notes": null
  },
  {
    "date": "2025-04-06T12:22:42",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 61.64516666666667,
    "actual_distance_miles": 6.365293455450001,
    "average_pace_minutes_per_mile": 9.684575754144646,
    "average_heart_rate_bpm": 151,
    "max_heart_rate_bpm": 165,
    "elevation_gain_feet": 124.67192,
    "notes": null
  },
  {
    "date": "2025-04-07T17:06:36",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 99.65566666666668,
    "actual_distance_miles": 6.3920434770000005,
    "average_pace_minutes_per_mile": 15.590580230758757,
    "average_heart_rate_bpm": 103,
    "max_heart_rate_bpm": 134,
    "elevation_gain_feet": 104.98688,
    "notes": null
  },
  {
    "date": "2025-04-08T19:49:58",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 37.26199999999999,
    "actual_distance_miles": 4.04087503236,
    "average_pace_minutes_per_mile": 9.221270071853175,
    "average_heart_rate_bpm": 161,
    "max_heart_rate_bpm": 181,
    "elevation_gain_feet": 65.6168,
    "notes": null
  },
  {
    "date": "2025-04-09T16:49:20",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 88.6535,
    "actual_distance_miles": 5.36010158875,
    "average_pace_minutes_per_mile": 16.539518613988506,
    "average_heart_rate_bpm": 96,
    "max_heart_rate_bpm": 106,
    "elevation_gain_feet": 26.24672,
    "notes": null
  },
  {
    "date": "2025-04-10T07:17:33",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 53.35816666666666,
    "actual_distance_miles": 5.270176777630001,
    "average_pace_minutes_per_mile": 10.124549691227216,
    "average_heart_rate_bpm": 149,
    "max_heart_rate_bpm": 163,
    "elevation_gain_feet": 82.021,
    "notes": null
  },
  {
    "date": "2025-04-11T11:59:01",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 54.24016666666667,
    "actual_distance_miles": 5.27210924144,
    "average_pace_minutes_per_mile": 10.288134062231942,
    "average_heart_rate_bpm": 156,
    "max_heart_rate_bpm": 167,
    "elevation_gain_feet": 98.4252,
    "notes": null
  },
  {
    "date": "2025-04-12T12:03:21",
    "workout_type": "Stair Climbing",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 59.54966666666667,
    "actual_distance_miles": 0.0,
    "average_pace_minutes_per_mile": null,
    "average_heart_rate_bpm": 135,
    "max_heart_rate_bpm": 153,
    "elevation_gain_feet": 1952.0998,
    "notes": null
  },
  {
    "date": "2025-04-13T18:16:52",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 40.122,
    "actual_distance_miles": 4.0412727098,
    "average_pace_minutes_per_mile": 9.928060509924263,
    "average_heart_rate_bpm": 143,
    "max_heart_rate_bpm": 152,
    "elevation_gain_feet": 32.8084,
    "notes": null
  },
  {
    "date": "2025-04-13T19:21:20",
    "workout_type": "Walking",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 14.004999999999999,
    "actual_distance_miles": 0.807160929,
    "average_pace_minutes_per_mile": 17.35093894764076,
    "average_heart_rate_bpm": 98,
    "max_heart_rate_bpm": 107,
    "elevation_gain_feet": 6.56168,
    "notes": null
  },
  {
    "date": "2025-04-14T19:29:35",
    "workout_type": "Running",
    "perceived_exertion": 5,
    "shin_and_knee_pain": 0,
    "shin_tightness": 0,
    "workout_adherence": 100,
    "actual_duration_minutes": 35.649166666666666,
    "actual_distance_miles": 4.05055599254,
    "average_pace_minutes_per_mile": 8.801055147076733,
    "average_heart_rate_bpm": 159,
    "max_heart_rate_bpm": 174,
    "elevation_gain_feet": 45.93176,
    "notes": null
  }
]
"""

with open(WORKOUT_DATA_FILE, "w") as f:
        f.write(workouts)


