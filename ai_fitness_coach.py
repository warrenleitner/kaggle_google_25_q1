from google import genai
import google.ai.generativelanguage as glm
from google.api_core import retry
from google.generativeai.types import GenerateContentResponse, ContentDict, PartDict
import os
import json
from datetime import datetime, date, timedelta
import typing
import typing_extensions
import sys
import time
from pathlib import Path
import langgraph
from langgraph.graph import StateGraph, END
import operator
from typing import Annotated, Sequence
from google.ai.generativelanguage import FunctionCall, Part as GlmPart, FunctionResponse

# --- Constants ---
KAGGLE_WORKING_DIR = Path("/kaggle/working/")
USER_DATA_FILE = KAGGLE_WORKING_DIR / "user_data.json"
DAILY_FEELING_FILE = KAGGLE_WORKING_DIR / "daily_feelings.json"
WORKOUT_DATA_FILE = KAGGLE_WORKING_DIR / "workout_data.json"
WEEKLY_SUMMARY_FILE = KAGGLE_WORKING_DIR / "weekly_summaries.json"
MONTHLY_STATS_FILE = KAGGLE_WORKING_DIR / "monthly_stats.json"
WORKOUT_PLAN_FILE = KAGGLE_WORKING_DIR / "current_workout_plan.json" # To store the latest plan

# --- API Key Configuration (IMPORTANT for Kaggle) ---
# Ensure you add your GOOGLE_API_KEY as a secret in Kaggle
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    print("Successfully configured Gemini API Key from Kaggle Secrets.")
except Exception as e:
    if os.environ.get("GOOGLE_API_KEY"):
        GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
        print("Successfully configured Gemini API Key from environment variable.")
    else:
        print(f"Error accessing Kaggle secrets or configuring API key: {e}")
        print("Please ensure GOOGLE_API_KEY is added as a Kaggle secret.")
        # Fallback for local testing (replace with your key)
        # GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
        # genai.configure(api_key=GOOGLE_API_KEY)
        # if GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        #      print("Using placeholder API Key. Replace for actual use.")
        sys.exit(1) # Exit if key isn't configured

# --- Gemini Client and Retry Policy ---
client = genai.Client(api_key=GOOGLE_API_KEY)

from google.api_core import retry

is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

if not hasattr(genai.models.Models.generate_content, '__wrapped__'):
  genai.models.Models.generate_content = retry.Retry(
      predicate=is_retriable)(genai.models.Models.generate_content)
  
# Apply the retry wrapper (do this carefully, ensuring it's applied only once)
# We'll call generate_content_with_retry instead of client.generate_content directly
# Note: Directly patching genai.models.Models.generate_content as in the prompt
# can be complex and might interfere with internal library mechanisms.
# Using a wrapper function is safer.

print("Gemini client and retry policy initialized.")


# --- JSON Type Definitions (from json_types.txt) ---

# Helper for date serialization/deserialization
class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, date):
            return obj.isoformat()
        return super().default(obj)

def date_decoder(json_dict):
    for key, value in json_dict.items():
        if isinstance(value, str):
            try:
                # Attempt to parse dates, handle potential errors gracefully
                json_dict[key] = datetime.strptime(value, '%Y-%m-%d').date()
            except ValueError:
                pass # Keep original string if not a valid date
    return json_dict

# --- 5. Static User Data ---
class StaticUserData(typing_extensions.TypedDict):
    """Stores baseline user information relevant for training plan personalization."""
    name: typing.Optional[str]
    age: typing.Optional[int]
    height_inches: typing.Optional[float]
    gender: typing.Optional[str]
    goal: typing.Optional[str]
    goal_time: typing.Optional[str] # Keep as string for flexibility (e.g., "sub 2:30")
    goal_date: typing.Optional[str]
    preferred_training_days: typing.Optional[typing.List[str]]
    long_run_day: typing.Optional[str]
    notes: typing.Optional[str]

# --- 1. Daily User Feeling ---
class DailyUserFeeling(typing_extensions.TypedDict):
    """Subjective user input regarding their physical and mental state on a specific day."""
    date: date
    overall_feeling: int
    energy_level: int
    shin_pain: int
    sleep_quality: int
    stress_level: typing.Optional[int]
    hydration_level: typing.Optional[int]
    nutrition_quality: typing.Optional[int]
    notes: typing.Optional[str]

# --- 2. Workout Data ---
class WorkoutData(typing_extensions.TypedDict):
    """Data for a single workout session (running or other types)."""
    date: date
    workout_type: str
    perceived_exertion: int # Made mandatory as it's often key
    shin_and_knee_pain: int
    shin_tightness: int
    workout_adherence: int
    actual_duration_minutes: float
    actual_distance_miles: typing.Optional[float]
    average_pace_minutes_per_mile: typing.Optional[float]
    average_heart_rate_bpm: typing.Optional[int]
    max_heart_rate_bpm: typing.Optional[int]
    elevation_gain_feet: typing.Optional[float]
    notes: typing.Optional[str]

# --- 3. Weekly User Summary (More Qualitative) ---
class WeeklyUserSummary(typing_extensions.TypedDict):
    """A qualitative summary of the user's training week."""
    week_start_date: date
    overall_summary: str
    key_achievements: typing.Optional[typing.List[str]]
    areas_for_focus: typing.Optional[typing.List[str]]
    total_workouts: int
    total_running_distance_miles: typing.Optional[float]
    total_workout_duration_minutes: typing.Optional[float]

# --- 4. Monthly User Stats (Primarily Quantitative Summary) ---
class MonthlyUserStats(typing_extensions.TypedDict):
    """Aggregated user statistics for a specific month."""
    month: str  # Format: "YYYY-MM"
    weight_pounds: float
    max_heart_rate_bpm: typing.Optional[int]
    resting_heart_rate_bpm: typing.Optional[int]
    vo2_max_estimated: typing.Optional[float]
    longest_run_distance_miles: float
    average_pace_minutes_per_mile: typing.Optional[float]
    comfortable_pace_minutes_per_mile: typing.Optional[float]
    comfortable_run_distance_miles: typing.Optional[float]
    average_heart_rate_bpm: typing.Optional[int]
    total_elevation_gain_feet: typing.Optional[float]
    monthly_summary_notes: typing.Optional[str]

print("JSON Type Definitions loaded.")


# --- Helper Functions ---

def load_json_data(filepath: Path) -> typing.Union[dict, list, None]:
    """Loads JSON data from a file, handling potential errors."""
    if not filepath.exists():
        return None
    try:
        with open(filepath, 'r') as f:
            # Use object_hook for date decoding
            return json.load(f, object_hook=date_decoder)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading JSON from {filepath}: {e}")
        return None

def save_json_data(filepath: Path, data: typing.Union[dict, list]):
    """Saves data to a JSON file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        with open(filepath, 'w') as f:
            # Use cls=DateEncoder for date encoding
            json.dump(data, f, indent=4, cls=DateEncoder)
        print(f"Data successfully saved to {filepath}")
    except IOError as e:
        print(f"Error saving JSON to {filepath}: {e}")
    except TypeError as e:
        print(f"Error serializing data for {filepath}: {e}")


def initialize_data_files():
    """Creates empty files if they don't exist (except user data)."""
    print("Initializing data files...")
    if not DAILY_FEELING_FILE.exists():
        save_json_data(DAILY_FEELING_FILE, [])
    if not WORKOUT_DATA_FILE.exists():
        save_json_data(WORKOUT_DATA_FILE, [])
    if not WEEKLY_SUMMARY_FILE.exists():
        save_json_data(WEEKLY_SUMMARY_FILE, [])
    if not MONTHLY_STATS_FILE.exists():
        save_json_data(MONTHLY_STATS_FILE, [])
    print("Data file initialization check complete.")

def check_last_updated(filepath: Path) -> typing.Optional[datetime]:
    """Gets the last modified time of a file."""
    if not filepath.exists():
        return None
    try:
        return datetime.fromtimestamp(filepath.stat().st_mtime)
    except OSError:
        return None

def upload_file_to_gemini(filepath: Path) -> typing.Optional[glm.File]:
    """Uploads a local file to the Gemini Files API."""
    print(f"Attempting to upload {filepath.name} to Gemini Files API...")
    try:
        # Ensure the file exists before attempting upload
        if not filepath.is_file():
            print(f"Error: File not found at {filepath}")
            return None

        # The Gemini API client handles the upload process
        # Use the genai.configure() setup client implicitly
        uploaded_file = client.files.upload(file=filepath)

        # Optional: Add a small delay to ensure file processing starts
        time.sleep(2)

        # Check file state (optional but good practice)
        file_info = client.files.get(name=uploaded_file.name)
        if file_info.state.name != "ACTIVE":
             print(f"Waiting for file {uploaded_file.name} to become active...")
             # Basic wait loop (consider a more robust polling mechanism for production)
             for _ in range(5): # Wait up to 10 seconds
                 time.sleep(2)
                 file_info = client.files.get(name=uploaded_file.name)
                 if file_info.state.name == "ACTIVE":
                     break
             if file_info.state.name != "ACTIVE":
                 print(f"Warning: File {uploaded_file.name} did not become active quickly. State: {file_info.state.name}")
                 # Depending on requirements, you might choose to return None or raise an error here
             else:
                 print(f"File {uploaded_file.name} is active.")
        else:
             print(f"File {uploaded_file.name} uploaded and active immediately.")


        print(f"Successfully uploaded {filepath.name} as {uploaded_file.name}")
        return uploaded_file # Return the File object needed for generate_content

    except Exception as e:
        print(f"Error uploading file {filepath.name} to Gemini: {e}")
        return None

def get_user_input_with_multimodal(prompt_text: str):
    """Gets text input and potentially file paths for multimodal input."""
    print(f"\n{prompt_text}")
    user_text = input("Your text response (or type 'skip'): ")
    if user_text.lower() == 'skip':
        return []

    files = []
    while True:
        file_path_str = input("Enter path to an image/video/audio file to upload (or press Enter to finish): ")
        if not file_path_str:
            break

        # Assume paths are relative to /kaggle/input/ or /kaggle/working/
        # It's crucial the user provides correct paths accessible by the Kaggle kernel
        file_path = Path(file_path_str)
        if not file_path.is_absolute():
           # Try resolving relative to common Kaggle dirs if not absolute
           possible_paths = [
               KAGGLE_WORKING_DIR / file_path,
               Path("/kaggle/input") / file_path, # Common for datasets
           ]
           found_path = None
           for p in possible_paths:
               if p.exists() and p.is_file():
                   found_path = p
                   break
           if not found_path:
                # Check if the original relative path exists from the current dir
                if (Path.cwd() / file_path).exists() and (Path.cwd() / file_path).is_file():
                    found_path = Path.cwd() / file_path
                else:
                    print(f"Error: Could not find file at '{file_path_str}' relative to common Kaggle directories or current directory.")
                    continue # Ask for another file path
           file_path = found_path # Use the resolved absolute path


        uploaded_file = upload_file_to_gemini(file_path)
        if uploaded_file:
            # IMPORTANT: Use Part.from_uri for uploaded files
            files.append(uploaded_file)
            print(f"Added {file_path.name} to input.")
        else:
            print(f"Skipping file {file_path.name} due to upload error.")

    return user_text, files


# --- Function Calling Definition ---
# Example function the LLM can call if it needs more info
def request_more_user_info(
    missing_fields: typing.List[str], # Which fields are unclear or missing
    clarification_needed: str # Specific question to ask the user
) -> str: # Return type annotation is important
    """
    Use this function ONLY when the user's input (text, images, audio, video)
    is insufficient to fully populate the required JSON fields.
    Ask the user for specific missing information or clarification.
    """
    print("\n--- AI Needs More Information ---")
    print(f"Missing or unclear fields: {', '.join(missing_fields)}")
    print(f"AI Request: {clarification_needed}")
    print("---------------------------------")
    # In a real app, you'd present this nicely and get structured user input.
    # For this prototype, we'll just get raw input.
    user_response = input("Please provide the requested information: ")
    return user_response # Return the user's text response back to the LLM

# --- Core AI Interaction Logic ---

# --- LangGraph State Definition ---
class StructuredOutputState(typing.TypedDict):
    system_prompt: str
    user_text: str
    user_files: typing.List[glm.File]
    output_schema: typing.Type[typing.TypedDict]
    allow_function_calling: bool
    messages: Annotated[Sequence[ContentDict], operator.add] # Accumulate messages
    max_fc_loops: int
    fc_loops: int
    final_output: typing.Optional[dict]
    error_message: typing.Optional[str]

# --- LangGraph Nodes ---

def prepare_initial_request(state: StructuredOutputState) -> StructuredOutputState:
    """Prepares the initial message list for the first Gemini call."""
    print("LangGraph: Preparing initial request...")
    initial_messages = [
        ContentDict(role="user", parts=[PartDict(text=state['user_text'])])
    ]
    for item in state.get('user_files', []):
        initial_messages.append(
             ContentDict(role="user", parts=[PartDict(file_data={
                 "mime_type": item.mime_type,
                 "file_uri": item.uri
             })])
        )
    return {"messages": initial_messages, "fc_loops": 0}


def call_gemini_node(state: StructuredOutputState) -> StructuredOutputState:
    """Calls the Gemini API with the current state."""
    loop_count = state['fc_loops']
    print(f"\nLangGraph: Calling Gemini (Loop {loop_count + 1})...")

    # Prepare tools and config for this call
    tools_list = [request_more_user_info] if state['allow_function_calling'] else None

    generation_config = genai.types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
        response_schema=state['output_schema'],
        # tools=tools_list, # Pass tools directly now
        # tool_config=tool_config,
        # Pass system instruction if supported by the model/API version being used
        # system_instruction=state['system_prompt'] # Pass directly to generate_content if needed
    )

    model_kwargs = {
        "model": "gemini-1.5-flash", # Or "gemini-2.0-flash" as used before
        "contents": state['messages'],
        "generation_config": generation_config,
        "tools": tools_list, # Pass tools here
    }
    # Add system instruction if using a model that supports it this way
    if "1.5" in model_kwargs["model"]: # Check model name
         model_kwargs["system_instruction"] = state['system_prompt']
    else:
         # For older models, prepend system prompt to contents if not directly supported
         # model_kwargs["contents"] = [ContentDict(role="system", parts=[PartDict(text=state['system_prompt'])])] + state['messages']
         print("Warning: System instruction might not be directly supported by the selected model in generate_content. Consider prepending to messages or upgrading model.")


    try:
        # Use the client defined earlier in the script
        response = client.generate_content(**model_kwargs)

        # Add the model's response (potentially including a function call) to messages
        # Ensure we handle cases where response might be blocked or empty
        if response.candidates and response.candidates[0].content.parts:
            # IMPORTANT: Convert the Part object from the SDK response to a ContentDict for LangGraph state
            # This assumes the response structure fits ContentDict expectations.
            # You might need to manually construct the ContentDict if the structure differs.
            response_content = response.candidates[0].content
            # Directly using response_content which is already a Content type object
            # Ensure it's serializable or convert to dict if needed by LangGraph state.
            # If response_content is already dict-like or a Content object compatible with LangGraph's state addition:
            return {"messages": [response_content]}
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
             error_msg = f"Gemini call blocked: {response.prompt_feedback.block_reason}"
             print(f"Error: {error_msg}")
             return {"error_message": error_msg, "final_output": None}
        else:
             error_msg = "Gemini response was empty or invalid."
             print(f"Error: {error_msg}")
             # print("Full Response:", response) # Debugging
             return {"error_message": error_msg, "final_output": None}

    except Exception as e:
        print(f"Error during Gemini API call in LangGraph: {e}")
        # print("Full Response (if available):", response) # Debugging
        return {"error_message": str(e), "final_output": None}


def process_response_node(state: StructuredOutputState) -> StructuredOutputState:
    """Processes the Gemini response, checking for function calls or final output."""
    print("LangGraph: Processing Gemini response...")
    if not state['messages']:
        print("Error: No messages found in state to process.")
        return {"final_output": None, "error_message": "State has no messages."}

    last_message = state['messages'][-1] # Get the latest message (model's response)

    # Ensure last_message is ContentDict or similar dict structure
    if not isinstance(last_message, dict) or 'parts' not in last_message:
        # If it's a Content object, convert to dict or access attributes
        # This depends on how LangGraph handles the objects internally when adding them
        # Assuming it might be a Content object, let's try accessing attributes
        if hasattr(last_message, 'parts') and last_message.parts:
            # Convert parts if necessary, assuming parts are list of Part objects
            parts_list = last_message.parts
        else:
            print("Error: Last message format is unexpected or has no parts.")
            return {"final_output": None, "error_message": "Unexpected last message format."}
    else: # It's already a dict
         parts_list = last_message.get('parts', [])

    if not parts_list:
        print("Error: Last message from model has no parts.")
        return {"final_output": None, "error_message": "Model response part is empty."}

    # Process the first part (assuming structure)
    # Part might be a Part object or a dict (PartDict)
    part_data = parts_list[0]

    # Check for function call
    # Access function_call based on whether part_data is an object or dict
    function_call = None
    if hasattr(part_data, 'function_call'): # Check if it's an object with the attribute
        function_call = part_data.function_call
    elif isinstance(part_data, dict) and 'function_call' in part_data: # Check if it's a dict with the key
        # If the value is a FunctionCall object, use it directly
        # If it's already a dict, ensure it has 'name' and 'args'
        fc_data = part_data['function_call']
        if isinstance(fc_data, FunctionCall):
             function_call = fc_data
        elif isinstance(fc_data, dict) and 'name' in fc_data:
             # Reconstruct FunctionCall object or use dict directly if FunctionResponse expects it
             # For simplicity, assuming we might get a dict representation here
             # Note: genai SDK FunctionCall expects args to be a MessageDict, not just dict
             # We might need proper conversion if using the SDK object strictly
              function_call = FunctionCall(name=fc_data.get('name'), args=fc_data.get('args', {})) # Basic reconstruction
        # function_call = part_data.get('function_call') # Returns dict if PartDict was used

    if function_call and state['allow_function_calling']:
        print(f"Gemini requested function call: {function_call.name}")

        if function_call.name == "request_more_user_info":
            # Check loop limit
            if state['fc_loops'] >= state['max_fc_loops']:
                 print("Error: Maximum function call loops reached.")
                 return {"final_output": None, "error_message": "Max function call loops reached."}

            # Extract args and call local function
            try:
                # FunctionCall object has args attribute (mappingproxy)
                args = dict(function_call.args)
                missing = args.get('missing_fields', [])
                clarification = args.get('clarification_needed', 'Could you provide more details?')
            except Exception as e:
                print(f"Error parsing function call arguments: {e}")
                missing = []
                clarification = "Could you provide more details or missing information?"

            function_response_text = request_more_user_info(
                missing_fields=missing,
                clarification_needed=clarification
            )

            # Prepare function response message using ContentDict and PartDict
            function_response_message = ContentDict(
                role="function", # Role should be 'function' for the response
                parts=[PartDict(function_response=FunctionResponse(
                    name=function_call.name,
                    response={"result": function_response_text}
                ))]
            )
            print("Added function response to history.")
            # We return the function response message to be added to the state
            # The conditional edge will route back to call_gemini
            return {"messages": [function_response_message], "fc_loops": state['fc_loops'] + 1}
        else:
            print(f"Warning: Received unhandled function call: {function_call.name}")
            # Treat as end state, try to extract JSON anyway? Or error out?
            # For now, let's try to extract JSON from potentially existing text part.
            pass # Fall through to JSON extraction attempt

    # Check for final JSON output (if no function call was processed)
    # Access text based on whether part_data is an object or dict
    text_content = None
    if hasattr(part_data, 'text'):
         text_content = part_data.text
    elif isinstance(part_data, dict) and 'text' in part_data:
         text_content = part_data.get('text')

    if text_content:
        try:
            json_string = text_content
            extracted_data = json.loads(json_string, object_hook=date_decoder)
            if isinstance(extracted_data, dict):
                print(f"Successfully extracted structured data for {state['output_schema'].__name__}.")
                return {"final_output": extracted_data, "error_message": None}
            else:
                print(f"Error: Gemini response was not a valid JSON object. Response: {extracted_data}")
                return {"final_output": None, "error_message": "Response was not a JSON object."}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response from Gemini: {e}")
            print(f"Raw response text: {text_content}")
            return {"final_output": None, "error_message": f"JSONDecodeError: {e}"}
        except Exception as e:
             print(f"Unexpected error processing final Gemini response text: {e}")
             return {"final_output": None, "error_message": f"Unexpected processing error: {e}"}
    else:
         # This might happen if the model only returned a function call that wasn't handled
         # Or if the response part genuinely didn't have text (e.g., only image)
         if not function_call: # Only error if we didn't expect a function call
             print("Error: No function call processed and no text part found in the final response.")
             print(f"Final response: {part_data}")
             return {"final_output": None, "error_message": "No text content in final response."}
         else: # A function call was present but unhandled / we decided to stop
              print("Info: No text content found, likely due to unhandled/terminal function call.")
              return {"final_output": None, "error_message": "Function call received, but no further processing or text output."}


# --- LangGraph Conditional Edge Logic ---

def should_continue(state: StructuredOutputState) -> str:
    """Determines whether to continue the loop (function call) or end."""
    if state.get("error_message"): # If an error occurred in previous steps
        print(f"LangGraph: Ending due to error: {state['error_message']}")
        return END
    if state.get("final_output") is not None: # If final JSON was extracted
         print("LangGraph: Ending with final output.")
         return END

    if not state['messages']: return END # Safety check

    last_message = state['messages'][-1]

    # Check the role of the last message (must be dict compatible)
    last_message_role = None
    if isinstance(last_message, dict):
        last_message_role = last_message.get('role')
    elif hasattr(last_message, 'role'):
        last_message_role = last_message.role


    # Check if the last message was a function *response* we just added
    if last_message_role == "function":
         # We just processed a function call and added the response, continue to call Gemini again
         if state['fc_loops'] < state['max_fc_loops']:
              print("LangGraph: Continuing loop after function response.")
              return "call_gemini"
         else:
              print("LangGraph: Ending loop, max retries reached after function call.")
              # Update error message maybe?
              # state["error_message"] = "Max function call loops reached." # State is immutable here
              return END # Max loops reached

    # If the last message was from the model and contained an *unhandled* function call,
    # or if it was just text (and processing failed to extract JSON above), we end.
    print("LangGraph: Ending (no function call processed or final JSON extracted).")
    return END


# --- Build the Graph ---

def build_structured_output_graph():
    workflow = StateGraph(StructuredOutputState)

    # Add nodes
    workflow.add_node("prepare_initial_request", prepare_initial_request)
    workflow.add_node("call_gemini", call_gemini_node)
    workflow.add_node("process_response", process_response_node)

    # Set entry point
    workflow.set_entry_point("prepare_initial_request")

    # Add edges
    workflow.add_edge("prepare_initial_request", "call_gemini")
    workflow.add_edge("call_gemini", "process_response")

    # Add conditional edge
    workflow.add_conditional_edges(
        "process_response",
        should_continue,
        {
            "call_gemini": "call_gemini", # Loop back if function call was handled
            END: END                   # End if finished or error
        }
    )

    # Compile the graph
    app = workflow.compile()
    print("LangGraph compiled successfully.")
    return app

# --- Replace the Original Function ---

structured_output_app = build_structured_output_graph() # Compile the graph once

def call_gemini_for_structured_output_langgraph(
    system_prompt: str,
    user_text: str, # Changed from user_content
    user_files: typing.List[glm.File],
    output_schema: typing.Type[typing.TypedDict],
    allow_function_calling: bool = False
) -> typing.Optional[typing.Dict]:
    """
    Calls the Gemini API via LangGraph to process input and return structured JSON data,
    handling function calls.
    """
    print(f"\nInvoking LangGraph for structured output ({output_schema.__name__})...")

    # Ensure user_text is a string
    if not isinstance(user_text, str):
        print(f"Warning: user_text was not a string ({type(user_text)}). Converting to string.")
        # Attempt a reasonable conversion, e.g., joining if it's a list
        if isinstance(user_text, list):
            user_text = "\n".join(map(str, user_text))
        else:
            user_text = str(user_text)


    initial_state = StructuredOutputState(
        system_prompt=system_prompt,
        user_text=user_text,
        user_files=user_files,
        output_schema=output_schema,
        allow_function_calling=allow_function_calling,
        messages=[], # Start with empty message list
        max_fc_loops=3,
        fc_loops=0,
        final_output=None,
        error_message=None
    )

    # Invoke the graph
    # Use configuration to handle recursion limits if necessary
    final_state = structured_output_app.invoke(initial_state, config={"recursion_limit": 10})


    # Return the final output from the state
    if final_state and final_state.get("final_output"):
        return final_state["final_output"]
    else:
        error_msg = final_state.get('error_message') if final_state else "Unknown error (final_state is None)"
        print(f"LangGraph execution finished without producing valid output. Error: {error_msg}")
        return None

# --- End of LangGraph Implementation ---


# --- Data Collection Functions ---

def collect_static_user_data() -> typing.Optional[StaticUserData]:
    """Collects initial user data if it doesn't exist."""
    print("\n--- Collecting Initial User Information ---")
    system_prompt = f"""
    You are an AI fitness coach assistant. Your task is to collect essential information
    from the user to personalize their fitness plan. Ask clarifying questions if needed,
    but primarily focus on gathering the data for the required JSON format.
    Ensure dates are in YYYY-MM-DD format.
    The required output format is JSON matching this structure: {StaticUserData.__annotations__}
    """
    prompt = (
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
    user_text, user_files = get_user_input_with_multimodal(prompt)
    if not user_text and not user_files:
        print("User skipped initial data collection.")
        return None

    # Allow function calling here if needed
    extracted_data = call_gemini_for_structured_output_langgraph(
        system_prompt=system_prompt,
        user_text=user_text,
        user_files=user_files,
        output_schema=StaticUserData,
        allow_function_calling=True # Allow requesting more info for user setup
    )

    if extracted_data:
        return typing.cast(StaticUserData, extracted_data)
    else:
        print("Could not extract user data.")
        return None


def collect_daily_feeling(target_date: date) -> typing.Optional[DailyUserFeeling]:
    """Prompts user for daily feeling input."""
    print(f"\n--- Logging Daily Feeling for {target_date.isoformat()} ---")
    system_prompt = f"""
    You are an AI fitness coach assistant. Collect the user's subjective feeling for {target_date.isoformat()}.
    Focus on the ratings (1-5 or 0-5 scales as defined) and any notes.
    The required output format is JSON matching this structure: {DailyUserFeeling.__annotations__}
    Set the 'date' field to {target_date.isoformat()}.
    Remind the user that hydration and nutrition reflect the *previous* day.
    """
    prompt = (
        f"How are you feeling today ({target_date.isoformat()})?\n"
        "- Overall feeling (1=Very Poor to 5=Excellent)?\n"
        "- Energy level (1=Very Low to 5=Very High)?\n"
        "- Shin pain (0=None to 5=Severe)?\n"
        "- Sleep quality last night (1=Very Poor to 5=Excellent)?\n"
        "- Stress level (Optional, 1=Very Low to 5=Very High)?\n"
        "- Hydration yesterday (Optional, 1=Poor to 5=Excellent)?\n"
        "- Nutrition yesterday (Optional, 1=Poor to 5=Excellent)?\n"
        "- Any other notes about how you feel?"
    )
    user_text, user_files = get_user_input_with_multimodal(prompt)
    if not user_text and not user_files:
        print("User skipped daily feeling log.")
        return None
    
    extracted_data = call_gemini_for_structured_output_langgraph(
        system_prompt=system_prompt,
        user_text=user_text,
        user_files=user_files,
        output_schema=DailyUserFeeling,
        allow_function_calling=True # Allow asking for clarification
    )

    if extracted_data:
        # Add/overwrite date just in case LLM missed it
        extracted_data['date'] = target_date
        # Basic validation
        if all(k in extracted_data for k in ['date', 'energy_level', 'sleep_quality']):
            return typing.cast(DailyUserFeeling, extracted_data)
        else:
            print("Error: Essential fields missing in extracted daily feeling.")
            print(f"Extracted: {extracted_data}")
            return None
    else:
        print("Could not extract daily feeling data.")
        return None


def collect_workout_data(target_date: date) -> typing.Optional[WorkoutData]:
    """Prompts user for workout data."""
    print(f"\n--- Logging Workout for {target_date.isoformat()} ---")
    system_prompt = f"""
    You are an AI fitness coach assistant. Collect details about the user's workout completed on {target_date.isoformat()}.
    Infer workout_type if not explicitly stated (e.g., 'run', 'strength', 'yoga', 'cycle').
    Extract quantitative data like duration, distance, pace, HR, elevation where provided.
    Capture perceived exertion (RPE 1-10) and specific pain/tightness ratings (0-5).
    Determine adherence percentage (0-100%).
    The required output format is JSON matching this structure: {WorkoutData.__annotations__}
    Set the 'date' field to {target_date.isoformat()}.
    """
    prompt = (
        f"Did you complete a workout today ({target_date.isoformat()})? If yes, please describe it:\n"
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
    user_text, user_files = get_user_input_with_multimodal(prompt)
    if not user_text and not user_files:
        print("User skipped workout log.")
        return None

    extracted_data = call_gemini_for_structured_output_langgraph(
        system_prompt=system_prompt,
        user_text=user_text,
        user_files=user_files,
        output_schema=WorkoutData,
        allow_function_calling=True # Allow asking for clarification
    )

    if extracted_data:
        extracted_data['date'] = target_date
        # Basic validation
        if all(k in extracted_data for k in ['date', 'workout_type', 'perceived_exertion', 'shin_and_knee_pain', 'shin_tightness']):
             return typing.cast(WorkoutData, extracted_data)
        else:
             print("Error: Essential fields missing in extracted workout data.")
             print(f"Extracted: {extracted_data}")
             # Fallback: Ask user if they want to save partial data? For now, return None.
             return None
    else:
        print("Could not extract workout data.")
        return None

# --- TODO: Implement collect_weekly_summary and collect_monthly_stats
# These would follow a similar pattern:
# 1. Define system prompt and user-facing prompt.
# 2. Call get_user_input_with_multimodal.
# 3. Call call_gemini_for_structured_output with the correct schema.
# 4. Validate and return the TypedDict or None.
# 5. For these summaries, the input might be less direct user text and more
#    about the AI *generating* the summary based on the week's/month's data,
#    possibly with some user confirmation/input. This is a more advanced step.

def collect_weekly_summary(week_start_date: date, workouts: list[WorkoutData], feelings: list[DailyUserFeeling]) -> typing.Optional[WeeklyUserSummary]:
    """Collects or generates a weekly summary."""
    print(f"\n--- Logging Weekly Summary for week starting {week_start_date.isoformat()} ---")

    # Prepare context from the week's data
    week_workouts = [w for w in workouts if week_start_date <= w['date'] < week_start_date + timedelta(days=7)]
    week_feelings = [f for f in feelings if week_start_date <= f['date'] < week_start_date + timedelta(days=7)]

    # Simple aggregation for context
    total_wk_workouts = len(week_workouts)
    total_wk_duration = sum(w.get('actual_duration_minutes', 0) for w in week_workouts)
    total_wk_distance = sum(w.get('actual_distance_miles', 0) or 0 for w in week_workouts if w.get('workout_type', '').lower() in ['run', 'easy run', 'tempo', 'intervals', 'long run'])

    system_prompt = f"""
    You are an AI fitness coach assistant. Generate a concise weekly summary for the user's training week starting {week_start_date.isoformat()}.
    Base the summary on the provided workout logs and daily feelings for the week.
    Highlight consistency, overall feeling, progress, key achievements (e.g., longest run, PR), and areas needing focus (e.g., recurring pain, missed workouts).
    Calculate quantitative totals.
    The required output format is JSON matching this structure: {WeeklyUserSummary.__annotations__}
    Set the 'week_start_date' field to {week_start_date.isoformat()}.
    Set 'total_workouts', 'total_running_distance_miles', and 'total_workout_duration_minutes' based on the calculations.
    """
    prompt = (
        f"Let's summarize your week starting {week_start_date.isoformat()}.\n"
        f"This week you logged {total_wk_workouts} workouts, totaling {total_wk_duration:.1f} minutes "
        f"and {total_wk_distance:.1f} running miles.\n"
        f"Workouts: {json.dumps(week_workouts, cls=DateEncoder, indent=2)}\n"
        f"Daily Feelings: {json.dumps(week_feelings, cls=DateEncoder, indent=2)}\n\n"
        "Based on this, what's your overall summary? Any key achievements or areas to focus on next week? "
        "(The AI will also generate a summary based on the data)."
    )

    user_text, user_files = get_user_input_with_multimodal(prompt) # Allow user to add their perspective

    # Combine AI context and user input
    full_context = [
        f"Context for week starting {week_start_date.isoformat()}:\n"
        f"Total Workouts: {total_wk_workouts}\n"
        f"Total Duration (min): {total_wk_duration:.1f}\n"
        f"Total Running Distance (miles): {total_wk_distance:.1f}\n"
        f"Workout Details:\n{json.dumps(week_workouts, cls=DateEncoder, indent=2)}\n\n"
        f"Daily Feeling Details:\n{json.dumps(week_feelings, cls=DateEncoder, indent=2)}\n\n"
        "User's input/perspective:"
    ]
    if user_text:
        full_context.append(user_text)
    else:
        full_context.append("User provided no additional input.")


    extracted_data = call_gemini_for_structured_output_langgraph(
        system_prompt=system_prompt,
        user_text=full_context_str, # Pass combined context string
        user_files=user_files,
        output_schema=WeeklyUserSummary,
        allow_function_calling=False # Probably not needed for summary generation
    )

    if extracted_data:
        # Ensure required fields are set by AI or calculations
        extracted_data['week_start_date'] = week_start_date
        extracted_data['total_workouts'] = total_wk_workouts
        extracted_data['total_running_distance_miles'] = total_wk_distance if total_wk_distance > 0 else None
        extracted_data['total_workout_duration_minutes'] = total_wk_duration if total_wk_duration > 0 else None

        if 'overall_summary' in extracted_data and extracted_data['overall_summary']:
            return typing.cast(WeeklyUserSummary, extracted_data)
        else:
             print("Error: Essential summary field missing in extracted weekly data.")
             print(f"Extracted: {extracted_data}")
             return None
    else:
        print("Could not extract weekly summary data.")
        return None

def collect_monthly_stats(year: int, month: int, user_data: StaticUserData, workouts: list[WorkoutData]) -> typing.Optional[MonthlyUserStats]:
    """Collects or generates monthly stats."""
    month_str = f"{year}-{month:02d}"
    print(f"\n--- Logging Monthly Stats for {month_str} ---")

    # Filter data for the month
    month_start = date(year, month, 1)
    next_month_start = date(year, month + 1, 1) if month < 12 else date(year + 1, 1, 1)
    month_workouts = [w for w in workouts if month_start <= w['date'] < next_month_start]

    # Basic aggregations (can be much more sophisticated)
    total_month_distance = sum(w.get('actual_distance_miles', 0) or 0 for w in month_workouts if w.get('workout_type', '').lower() in ['run', 'easy run', 'tempo', 'intervals', 'long run'])
    running_workouts = [w for w in month_workouts if w.get('actual_distance_miles', 0) and w.get('actual_distance_miles', 0) > 0]
    longest_run = max(w.get('actual_distance_miles', 0) or 0 for w in running_workouts) if running_workouts else 0.0

    # TODO: More complex calculations for avg pace, comfortable pace, HR, elevation etc.
    # These require more careful filtering and averaging.
    # For prototype, we might ask user or make simpler estimates.

    system_prompt = f"""
    You are an AI fitness coach assistant. Generate a quantitative monthly stats summary for {month_str}.
    Base the summary on the provided workout logs and user data.
    Estimate or ask the user for values like current weight, resting/max HR if not available.
    Calculate longest run, average pace (if possible), total elevation etc.
    The required output format is JSON matching this structure: {MonthlyUserStats.__annotations__}
    Set the 'month' field to '{month_str}'.
    """
    prompt = (
        f"Let's update your stats for {month_str}.\n"
        f"Workouts this month: {len(month_workouts)}\n"
        f"Total running distance: {total_month_distance:.1f} miles\n"
        f"Longest run: {longest_run:.1f} miles\n"
        f"Workout Details: {json.dumps(month_workouts, cls=DateEncoder, indent=2)}\n\n"
        "Please provide or confirm the following for this month:\n"
        "- Current weight (pounds)?\n"
        "- Resting Heart Rate (BPM) (optional)?\n"
        "- Max Heart Rate (BPM) (optional, if tested/known)?\n"
        "- Estimated VO2 Max (optional)?\n"
        "- Any notes summarizing the month?\n"
        "(The AI will also generate stats based on the data)."
    )

    user_text, user_files = get_user_input_with_multimodal(prompt)

    # Combine context
    full_context = [
        f"Context for month {month_str}:\n"
        f"User Profile: {json.dumps(user_data, cls=DateEncoder)}\n"
        f"Workouts Logged: {len(month_workouts)}\n"
        f"Calculated Longest Run (miles): {longest_run:.1f}\n"
        # Add more calculated fields here as developed
        f"Workout Details:\n{json.dumps(month_workouts, cls=DateEncoder, indent=2)}\n\n"
        "User's input/perspective:"
    ]
    if user_text:
        full_context.extend(user_text)
    else:
        full_context.append("User provided no additional input.")

    extracted_data = call_gemini_for_structured_output_langgraph(
        system_prompt=system_prompt,
        user_text=full_context_str, # Pass combined context string
        user_files=user_files,
        output_schema=MonthlyUserStats,
        allow_function_calling=True # Allow asking for weight etc.
    )

    if extracted_data:
        extracted_data['month'] = month_str
        # Ensure calculated fields are present
        extracted_data['longest_run_distance_miles'] = longest_run

        # Basic validation
        if 'month' in extracted_data and 'weight_pounds' in extracted_data: # Weight is often key
            return typing.cast(MonthlyUserStats, extracted_data)
        else:
             print("Error: Essential fields missing in extracted monthly stats.")
             print(f"Extracted: {extracted_data}")
             return None
    else:
        print("Could not extract monthly stats data.")
        return None


# --- Workout Planning and Q&A ---

def plan_workout(all_data: dict) -> typing.Optional[dict]:
    """Generates the next workout plan based on all available data."""
    print("\n--- Generating Workout Plan ---")

    # Prune data slightly for prompt length if necessary (e.g., only last few weeks/months)
    # For prototype, send all
    context_json = json.dumps(all_data, cls=DateEncoder, indent=2)

    # Check context length (approximate)
    if len(context_json) > 250000: # Example limit, adjust based on model/API limits
        print("Warning: Context data is very large, potentially trimming...")
        # Implement trimming logic here if needed (e.g., keep last N entries)
        # context_json = json.dumps(trimmed_data, cls=DateEncoder, indent=2)
        pass # Placeholder for trimming logic


    system_prompt = """
    You are an expert AI running coach. Analyze the provided user data (static info, goals,
    daily feelings, workout logs, weekly summaries, monthly stats) to generate the
    next logical workout plan for the user.

    Consider:
    - User's goal, goal date, and current fitness level (inferred from data).
    - Recent workout performance, perceived exertion, and adherence.
    - Subjective feedback: energy levels, sleep, stress, pain (especially shin issues).
    - Weekly/monthly trends and summaries.
    - User's preferred training days and long run day.
    - General principles of training periodization (base building, intensity, tapering).

    Output a structured workout plan for the next session or few sessions (e.g., for the next day or two).
    The plan should include:
    - Date(s)
    - Workout Type (e.g., Easy Run, Tempo, Intervals, Strength, Rest)
    - Description (e.g., 'Run 3 miles at easy pace (RPE 3-4)', 'Warmup, 3x800m @ target pace w/ recovery, Cooldown')
    - Target Duration/Distance
    - Target Intensity (e.g., RPE, pace range, heart rate zone)
    - Specific notes (e.g., 'Focus on form', 'Monitor shin pain closely')

    Be prepared to revise the plan based on user feedback. Output the plan as a JSON object
    with a key like "workout_plan" containing a list of workout objects.
    Example Workout Object:
    { "date": "YYYY-MM-DD", "type": "Easy Run", "description": "Run 4 miles at conversational pace", "target_duration_minutes": 40, "target_intensity": "RPE 3-4", "notes": "Focus on hydration before." }
    """

    # Simple interaction loop
    while True:
        print("Generating initial workout plan...")
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[system_prompt, f"User Data:\n{context_json}"],
                generation_config=genai.types.GenerateContentConfig(
                    temperature=0.5, # Allow some creativity in planning
                    # No JSON schema here initially, as format is less rigid
                    # response_mime_type="application/json" # Could enforce top-level JSON
                ),
            )

            if response.candidates and response.candidates[0].content.parts:
                 plan_text = response.candidates[0].content.parts[0].text
                 print("\n--- Proposed Workout Plan ---")
                 print(plan_text)
                 print("-----------------------------")

                 # Attempt to parse as JSON if possible (if AI followed instructions)
                 try:
                     # Look for JSON block within the text
                     json_start = plan_text.find('{')
                     json_end = plan_text.rfind('}') + 1
                     if json_start != -1 and json_end != -1:
                         plan_json_str = plan_text[json_start:json_end]
                         workout_plan_data = json.loads(plan_json_str, object_hook=date_decoder)
                         print("(Successfully parsed plan as JSON)")
                     else:
                          workout_plan_data = {"plan_text": plan_text} # Store raw text if not JSON
                          print("(Plan stored as raw text)")

                 except json.JSONDecodeError:
                     workout_plan_data = {"plan_text": plan_text} # Store raw text if parse fails
                     print("(Could not parse plan as JSON, stored as raw text)")


                 feedback = input("What do you think? Enter feedback, ask questions, or type 'accept': ")

                 if feedback.lower() == 'accept':
                     print("Workout plan accepted!")
                     save_json_data(WORKOUT_PLAN_FILE, workout_plan_data) # Save the accepted plan
                     return workout_plan_data
                 elif feedback.lower() in ['quit', 'exit']:
                     print("Exiting planning phase.")
                     return None
                 else:
                     # Send feedback back to the LLM to refine the plan
                     context_json = json.dumps({"user_data": all_data, "previous_plan": plan_text, "user_feedback": feedback}, cls=DateEncoder)
                     # Shorten context if needed for follow-up
                     if len(context_json) > 250000:
                           context_json = json.dumps({"user_feedback": feedback, "previous_plan_summary": plan_text[:500]}, cls=DateEncoder) # Simplified context

                     print("Asking AI to revise plan based on feedback...")
                     # Loop continues

            else:
                 print("Error: Failed to generate workout plan.")
                 if response:
                     try: print(f"Safety Feedback: {response.prompt_feedback}")
                     except AttributeError: pass
                 return None

        except Exception as e:
            print(f"Error during workout planning: {e}")
            return None


def workout_q_and_a(all_data: dict, workout_plan: dict):
    """Handles user questions about the workout plan and fitness data."""
    print("\n--- Workout Q&A ---")
    print("Ask any questions about your plan, your data, or general fitness.")

    context_json = json.dumps({"user_data": all_data, "accepted_plan": workout_plan}, cls=DateEncoder, indent=2)
    # Trim if needed
    if len(context_json) > 250000:
        print("Warning: Q&A context data is very large, potentially trimming...")
        # Implement trimming
        pass

    system_prompt = """
    You are an AI fitness coach assistant. The user has accepted a workout plan.
    Your role now is to answer the user's questions clearly and concisely.
    Refer to the user's complete data history (profile, goals, logs, summaries, stats)
    and the accepted workout plan provided below.
    Explain the reasoning behind the plan, define terms, offer encouragement,
    or provide general fitness advice based on their context.
    Be helpful and supportive.
    """

    # Use the Chat API for conversation history
    chat_client = genai.GenerativeModel(model_name='gemini-2.0-flash', system_instruction=system_prompt)
    chat = chat_client.start_chat(history=[
        {'role': 'user', 'parts': [f"Here is my data and the workout plan we agreed on:\n{context_json}"]},
        {'role': 'model', 'parts': ["Okay, I have reviewed your data and the workout plan. How can I help you?"]}
    ])


    while True:
        user_question = input("\nYour question (or type 'done'): ")
        if user_question.lower() == 'done':
            print("Exiting Q&A.")
            break
        if not user_question:
            continue

        try:
            print("Asking AI...")
            response = chat.send_message(user_question, stream=False) # Use the chat object

            if response.candidates and response.candidates[0].content.parts:
                 answer = response.candidates[0].content.parts[0].text
                 print(f"\nAI Coach: {answer}")
            else:
                 print("AI Coach: Sorry, I couldn't generate a response for that.")
                 try: print(f"Safety Feedback: {response.prompt_feedback}")
                 except AttributeError: pass

        except Exception as e:
            print(f"Error during Q&A: {e}")
            # Attempt to continue the loop


# --- Main Application Logic ---

def main():
    """Main function to run the AI fitness coach CLI."""
    print("--- AI Fitness Coach Prototype ---")

    # 1. Load or Initialize Data
    user_data: typing.Optional[StaticUserData] = load_json_data(USER_DATA_FILE)
    daily_feelings: list[DailyUserFeeling] = load_json_data(DAILY_FEELING_FILE) or []
    workout_data: list[WorkoutData] = load_json_data(WORKOUT_DATA_FILE) or []
    weekly_summaries: list[WeeklyUserSummary] = load_json_data(WEEKLY_SUMMARY_FILE) or []
    monthly_stats: list[MonthlyUserStats] = load_json_data(MONTHLY_STATS_FILE) or []

    if user_data is None:
        user_data = collect_static_user_data()
        if user_data:
            save_json_data(USER_DATA_FILE, user_data)
        else:
            print("User data collection failed or skipped. Exiting.")
            return # Can't proceed without basic user data

    initialize_data_files() # Ensure list files exist even if empty

    # 2. Check for Missing Data & Prompt User
    today = date.today()
    last_daily_log_date = daily_feelings[-1]['date'] if daily_feelings else None
    last_workout_log_date = workout_data[-1]['date'] if workout_data else None

    # --- Daily Checks ---
    if last_daily_log_date is None or last_daily_log_date < today:
        print(f"\nLooks like you haven't logged your feeling for today ({today.isoformat()}).")
        new_feeling = collect_daily_feeling(today)
        if new_feeling:
            daily_feelings.append(new_feeling)
            save_json_data(DAILY_FEELING_FILE, daily_feelings)
    else:
         print(f"\nDaily feeling log is up-to-date ({last_daily_log_date.isoformat()}).")


    # Simple check: prompt for workout if feeling logged but workout isn't for today
    if (last_daily_log_date == today) and (last_workout_log_date is None or last_workout_log_date < today):
         # More robust check: Did they *actually* workout today? Maybe ask first.
         ask_workout = input(f"Did you complete a workout today ({today.isoformat()})? (yes/no): ")
         if ask_workout.lower() == 'yes':
             new_workout = collect_workout_data(today)
             if new_workout:
                 workout_data.append(new_workout)
                 save_json_data(WORKOUT_DATA_FILE, workout_data)
         else:
              print("Okay, no workout logged for today.")
    elif last_workout_log_date == today:
         print(f"Workout log is up-to-date ({last_workout_log_date.isoformat()}).")


    # --- Weekly Check ---
    last_monday = today - timedelta(days=today.weekday())
    last_summary_date = weekly_summaries[-1]['week_start_date'] if weekly_summaries else None

    if last_summary_date is None or last_summary_date < last_monday:
         # Check if the *previous* week needs summarizing
         prev_week_start = last_monday - timedelta(days=7)
         if last_summary_date is None or last_summary_date < prev_week_start:
             print(f"\nLooks like the summary for the week starting {prev_week_start.isoformat()} is missing.")
             # Need workouts/feelings from that week to generate summary
             prev_week_workouts = [w for w in workout_data if prev_week_start <= w['date'] < last_monday]
             prev_week_feelings = [f for f in daily_feelings if prev_week_start <= f['date'] < last_monday]
             if prev_week_workouts or prev_week_feelings: # Only summarize if there's data
                 new_summary = collect_weekly_summary(prev_week_start, workout_data, daily_feelings)
                 if new_summary:
                     weekly_summaries.append(new_summary)
                     save_json_data(WEEKLY_SUMMARY_FILE, weekly_summaries)
             else:
                  print(f"(No data found for week {prev_week_start.isoformat()} to summarize)")

    else:
         print(f"\nWeekly summary is up-to-date (last summary for week starting {last_summary_date.isoformat()}).")


    # --- Monthly Check ---
    current_month_str = f"{today.year}-{today.month:02d}"
    last_month_stat = monthly_stats[-1]['month'] if monthly_stats else None

    if last_month_stat is None or last_month_stat < current_month_str:
         # Check if the *previous* month needs stats
         prev_month_date = today.replace(day=1) - timedelta(days=1)
         prev_month_str = f"{prev_month_date.year}-{prev_month_date.month:02d}"

         if last_month_stat is None or last_month_stat < prev_month_str:
             print(f"\nLooks like the stats for {prev_month_str} are missing.")
             new_stats = collect_monthly_stats(prev_month_date.year, prev_month_date.month, user_data, workout_data)
             if new_stats:
                 monthly_stats.append(new_stats)
                 save_json_data(MONTHLY_STATS_FILE, monthly_stats)
    else:
         print(f"\nMonthly stats are up-to-date (last stats for {last_month_stat}).")


    print("\n--- All data checks complete ---")

    # 3. Workout Planning Phase
    all_current_data = {
        "user_profile": user_data,
        "daily_feelings": daily_feelings,
        "workout_logs": workout_data,
        "weekly_summaries": weekly_summaries,
        "monthly_stats": monthly_stats
    }

    accepted_plan = plan_workout(all_current_data)

    # 4. Q&A Phase
    if accepted_plan:
        workout_q_and_a(all_current_data, accepted_plan)
    else:
        print("\nNo workout plan was accepted or generated.")

    print("\n--- AI Fitness Coach session finished ---")


if __name__ == "__main__":
    main()