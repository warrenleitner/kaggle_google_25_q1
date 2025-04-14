from ast import List
from pprint import pprint
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
import readline

# --- Constants ---
if os.environ.get("KAGGLE_WORKING_DIR"):
    WORKING_DIR = Path("/kaggle/working/")
else:
    WORKING_DIR = Path.cwd()


USER_DATA_FILE = WORKING_DIR / "user_data.json"
DAILY_FEELING_FILE = WORKING_DIR / "daily_feelings.json"
WORKOUT_DATA_FILE = WORKING_DIR / "workout_data.json"
WEEKLY_SUMMARY_FILE = WORKING_DIR / "weekly_summaries.json"
MONTHLY_STATS_FILE = WORKING_DIR / "monthly_stats.json"
WORKOUT_PLAN_FILE = WORKING_DIR / "current_workout_plan.json" # To store the latest plan

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
class StaticUserData(typing.TypedDict):
    """Stores baseline user information relevant for training plan personalization."""
    name: str  # User's name
    age: int  # User's age in years
    height_inches: typing.Optional[float] # User's height in inches (optional)
    gender: typing.Optional[str] # User's reported gender (optional, e.g., "Male", "Female", "Non-binary", "Prefer not to say")
     # --- Goal --- 
    goal: str  # User's primary goal (e.g., "5K", "10K", "Half Marathon", "Marathon", "Ultra", "Other")
    goal_time: typing.Optional[str]  # Target time for the goal (e.g., 2.5 for 2 hours 30 minutes)
    goal_date: datetime  # Date of the user would like to be ready for their goal. (e.g., "2024-05-15")
     # --- Training Preferences (Optional) ---
    preferred_training_days: typing.Optional[typing.List[str]] # Days user prefers to train (e.g., ["Monday", "Wednesday", "Friday", "Sunday"])
    long_run_day: typing.Optional[str] # Preferred day for the weekly long run (e.g., "Saturday")
     # --- Other Optional Data ---
    notes: typing.Optional[str]  # Optional free-text notes for additional context

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
   "required": []
}
"""

# --- 1. Daily User Feeling ---
class DailyUserFeeling(typing.TypedDict):
    """Subjective user input regarding their physical and mental state on a specific day."""
    date: datetime  # The date this feeling log applies to
    overall_feeling: int  # Rating scale, e.g., 1 (Very Poor) to 5 (Excellent)
    energy_level: int  # Rating scale, e.g., 1 (Very Low) to 5 (Very High)
    shin_pain: int  # Rating scale, e.g., 0 (None) to 5 (Severe)
    sleep_quality: int  # Rating scale, e.g., 1 (Very Poor) to 5 (Excellent)
    stress_level: typing.Optional[int]  # Optional: 1 (Very Low) to 5 (Very High)
    hydration_level: typing.Optional[int]  # 1 (Poor) to 5 (Excellent). Note that this represents the previous day
    nutrition_quality: typing.Optional[int]  # 1 (Poor) to 5 (Excellent). Note that this represents the previous day
    notes: typing.Optional[str]  # Optional free-text notes

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

# --- 2. Workout Data ---
class WorkoutData(typing.TypedDict):
    """Data for a single workout session (running or other types)."""
    date: datetime  # Date of the workout
    workout_type: str  # E.g., "Easy Run", "Tempo", "Strength", "Yoga", "Cycling", "Other"
    perceived_exertion: int  # Optional: RPE scale 1-10
    shin_and_knee_pain: int  # Optional: 0 (None) to 5 (Severe) (Specific to this user's shin pain)
    shin_tightness: int  # Optional: 0 (None) to 5 (Severe) (Specific to this user's shin tightness)
    workout_adherence: int # Percent adherence to the planned workout (0-100%)
    actual_duration_minutes: float # Total workout time in minutes
    actual_distance_miles: typing.Optional[float]  # Distance in miles (None if not applicable)
    average_pace_minutes_per_mile: typing.Optional[float]  # Pace in min/mile (None if not applicable)
    average_heart_rate_bpm: typing.Optional[int]
    max_heart_rate_bpm: typing.Optional[int]
    elevation_gain_feet: typing.Optional[float]  # Elevation gain in feet (None if not applicable)
    notes: typing.Optional[str]  # User comments, could include exercises/sets/reps for strength

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

# --- 3. Weekly User Summary (More Qualitative) ---
class WeeklyUserSummary(typing.TypedDict):
    """A qualitative summary of the user's training week."""
    week_start_date: datetime  # E.g., the Monday of the week
    overall_summary: str  # Text summary of the week (consistency, feeling, progress)
    key_achievements: typing.Optional[typing.List[str]]  # Bullet points of successes
    areas_for_focus: typing.Optional[typing.List[str]]  # Bullet points for improvement areas
    total_workouts: int # Total number of workouts logged this week
    # Optional quantitative context if available/relevant
    total_running_distance_miles: typing.Optional[float]
    total_workout_duration_minutes: typing.Optional[float]

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

# --- 4. Monthly User Stats (Primarily Quantitative Summary) ---
class MonthlyUserStats(typing.TypedDict):
    """Aggregated user statistics for a specific month."""
    month: str  # Format: "YYYY-MM"
    # --- Optional Physiological Estimates ---
    weight_pounds: float # User's current weight in pounds
    max_heart_rate_bpm: typing.Optional[int] # Max HR (optional, estimated or tested)
    resting_heart_rate_bpm: typing.Optional[int] # Resting HR (optional)
    vo2_max_estimated: typing.Optional[float] # Estimated VO2 Max (optional)
    longest_run_distance_miles: float
    average_pace_minutes_per_mile: typing.Optional[float] # Avg pace for runs this month
    comfortable_pace_minutes_per_mile: typing.Optional[float] # Comfortable pace for runs this month
    comfortable_run_distance_miles: typing.Optional[float] # Avg comfortable run distance
    average_heart_rate_bpm: typing.Optional[int] # Avg HR across workouts with HR data
    total_elevation_gain_feet: typing.Optional[float] # Total elevation for runs
    monthly_summary_notes: typing.Optional[str] # Optional field for brief LLM-generated or user notes

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
      "description": "Max HR (optional, estimated or tested)",
      "type": ["integer", "null"],
      "minimum": 0
    },
    "resting_heart_rate_bpm": {
      "description": "Resting HR (optional)",
      "type": ["integer", "null"],
      "minimum": 0
    },
    "vo2_max_estimated": {
      "description": "Estimated VO2 Max (optional)",
      "type": ["number", "null"],
      "minimum": 0
    },
    "longest_run_distance_miles": {
      "description": "Longest run distance this month",
      "type": "number",
      "minimum": 0
    },
    "average_pace_minutes_per_mile": {
      "description": "Avg pace for runs this month",
      "type": ["number", "null"],
      "minimum": 0
    },
    "comfortable_pace_minutes_per_mile": {
      "description": "Comfortable pace for runs this month",
      "type": ["number", "null"],
      "minimum": 0
    },
    "comfortable_run_distance_miles": {
      "description": "Avg comfortable run distance",
      "type": ["number", "null"],
      "minimum": 0
    },
    "average_heart_rate_bpm": {
      "description": "Avg HR across workouts with HR data",
      "type": ["integer", "null"],
      "minimum": 0
    },
    "total_elevation_gain_feet": {
      "description": "Total elevation for runs",
      "type": ["number", "null"]
    },
    "monthly_summary_notes": {
      "description": "Optional field for brief LLM-generated or user notes",
      "type": ["string", "null"]
    }
  },
  "required": [
    "month",
    "weight_pounds",
    "longest_run_distance_miles"
  ]
}
"""

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
#  --- Get additional input from user ---
def get_user_input_with_multimodal(prompt_text: str):
    """Gets text input and potentially file paths for multimodal input."""
    print(f"\n{prompt_text}")
    user_text = input("Your text response (press enter to instead upload a file or type 'skip' to ignore this update): ")
    if user_text.lower() == 'skip':
        return []

    files = []
    while True:
        file_path_str = input("Enter path to an image/video/audio file to upload (or press Enter to finish): ")
        if not file_path_str:
            break

        file_path = Path(file_path_str)
        if not file_path.is_absolute():
           # Try resolving relative to current directory
           if (Path.cwd() / file_path).exists() and (Path.cwd() / file_path).is_file():
               file_path = Path.cwd() / file_path
           else:
               print(f"Error: Could not find file at '{file_path_str}'.")
               continue  # Ask for another file path

        try:
            # The Gemini API client handles the upload process
            uploaded_file = client.files.upload(file=file_path)
            files.append(uploaded_file)
            print(f"Added {file_path.name} to input.")
        except Exception as e:
            print(f"Error uploading file {file_path.name} to Gemini: {e}")

    return user_text, files

def parse_input_into_messages(user_text: str, user_files: list[glm.File]) -> list[ContentDict]:
    """Prepares the initial message list for the first Gemini call."""
    
    messages = [
        ContentDict(role="user", parts=[genai.types.Part.from_text(text=user_text)])
    ]
    for item in user_files:
        messages.append(
            ContentDict(role="user", parts=[genai.types.Part.from_uri(file_uri=item.uri, mime_type=item.mime_type)])
        )
    return messages



# --- LangGraph Nodes ---

def call_gemini_for_structured_output(
    system_prompt: str,
    user_text: str, # Changed from user_content
    user_files: typing.List[glm.File],
    output_schema: typing.Type[typing.TypedDict],
):
    """Calls Gemini with function calling enabled to gather data."""

    messages = parse_input_into_messages(user_text, user_files)
    sufficient_info = False

    # --- Function that the LLM can call to request more information ---
    def request_more_user_info(
        missing_fields: list[str],  # Which fields are unclear or missing
        clarification_needed: str  # Specific question to ask the user
    ):
        """
        Called by the Gemini model when user input lacks required information.

        This function prompts the user for specific missing details identified by the model.
        It uses `get_user_input_with_multimodal` to gather text and optional file uploads
        from the user and appends the new information to the global `messages` list
        for subsequent model processing.

        Args:
            missing_fields: A list of field names (strings) that the model
                             determined were missing or unclear from the user's input.
            clarification_needed: The specific question the model wants to ask the user
                                   to obtain the missing information.
        """
        prompt = "\n--- AI Needs More Information ---"
        prompt += f"\nMissing or unclear fields: {', '.join(missing_fields)}"
        prompt += f"\nAI Request: {clarification_needed}"
        prompt += "\n---------------------------------"
        # Note: Uses global `messages` which might be refactored later
        new_user_text, new_user_files = get_user_input_with_multimodal(prompt)
        # Append the new user input and any uploaded files to the message history
        if new_user_text or new_user_files:
             messages.append(parse_input_into_messages(new_user_text, new_user_files))
        else:
            # Handle case where user provides no input to the clarification request
            # Maybe append a message indicating this? For now, just proceed.
            print("User provided no further input for clarification.")

    def proceed_to_json_generation():
        """
        Called by the Gemini model when it determines it has sufficient information.

        This function signals that the information gathering phase is complete.
        It sets the global `sufficient_info` flag to True, which terminates
        the loop in `call_gemini_for_structured_output` and allows the process
        to proceed to the final JSON generation step.
        """
        # Note: Uses global `sufficient_info` which might be refactored later
        nonlocal sufficient_info
        sufficient_info = True

    model_name = "gemini-2.0-flash"  # @param ["gemini-2.0-flash-lite","gemini-2.0-flash","gemini-2.5-pro-exp-03-25"] {"allow-input":true}

    max_loops = 3
    loop_count = 0
    while ((not sufficient_info) and (loop_count < max_loops)):
        loop_count += 1
        print(f"Calling Gemini ({client.models}) for structured output ({output_schema.__name__})...")
        print(f"Sufficient info: {sufficient_info}")
        print(f"Not sufficient info: {not sufficient_info}")
        print(f"Loop count: {loop_count}")
        print(f"Loop Status: {(not sufficient_info) and (loop_count < max_loops)}")
        try:
            response = client.models.generate_content(
                model=model_name,
                config=genai.types.GenerateContentConfig(
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

            tool_call = response.candidates[0].content.parts[0].function_call
            messages.append(genai.types.ContentDict(role="user", parts=[genai.types.Part.from_text(text=f"NOTE: System replied with - Function call: {tool_call.name} with args: {tool_call.args}")]))
            if tool_call.name == "proceed_to_json_generation":
                print(f"Proceeding to JSON generation.")
                proceed_to_json_generation()
            elif tool_call.name == "request_more_user_info":
                print(f"Requesting more user info.")
                request_more_user_info(**tool_call.args)
            else:
                print(f"Unknown tool call: {tool_call.name}")
                messages.append(genai.types.ContentDict(role="user", parts=[genai.types.Part.from_text(text=f"Unknown tool call: {tool_call.name}")]))
        except Exception as e:
            print(f"Error: {e}")
            print(f"Messages: {messages}")
            print(f"Response: {response}")
            sys.exit()

    print(messages)


    print(f"\nAttempting to generate JSON...")
    # print("Messages:", messages) # Debugging: See what's being sent
    # print("Tools:", tools_list) # Debugging: See tools being sent

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=genai.types.GenerateContentConfig(
            temperature=0.2, # Lower temp for more deterministic JSON extraction
            response_mime_type="application/json",
            response_schema=output_schema
        ),
        contents=messages
    )
    pprint(response)

    # print("Raw Gemini Response:", response) # Debugging: See the full response

    # --- Process Final Response ---
    try:
        if response.candidates and response.candidates[0].content.parts:
             # Assuming the JSON is in the first part of the last message
             final_part = response.candidates[0].content.parts[0]
             if final_part.text: # Check if text part exists (where JSON is expected)
                 # The API should directly return JSON parseable text when mime_type is set
                 json_string = final_part.text
                 # print(f"Received JSON string: {json_string}") # Debugging
                 extracted_data = json.loads(json_string, object_hook=date_decoder)

                 # Basic validation (check if it's a dict)
                 if isinstance(extracted_data, dict):
                     print(f"Successfully extracted structured data for {output_schema.__name__}.")
                     return extracted_data
                 else:
                     print(f"Error: Gemini response was not a valid JSON object for {output_schema.__name__}. Response: {extracted_data}")
                     return None
             else:
                  print(f"Error: Gemini response part did not contain text for {output_schema.__name__}.")
                  # print("Response Part:", final_part) # Debugging
                  return None

        else:
            print(f"Error: No valid candidates or content parts in Gemini response for {output_schema.__name__}.")
            # print("Full Response:", response) # Debugging
            return None

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from Gemini for {output_schema.__name__}: {e}")
        print(f"Raw response text: {response.candidates[0].content.parts[0].text}")
        return None
    except Exception as e:
        print(f"Unexpected error processing final Gemini response: {e}")
        # print("Full Response:", response) # Debugging
        return None


# --- Data Collection Functions ---

def collect_static_user_data() -> typing.Optional[StaticUserData]:
    """Collects initial user data if it doesn't exist."""
    print("\n--- Collecting Initial User Information ---")
    system_prompt = f"""
    You are an AI fitness coach assistant. Your task is to collect essential information
    from the user to personalize their fitness plan. Ask clarifying questions if needed,
    but primarily focus on gathering the data for the required JSON format.
    Ensure dates are in YYYY-MM-DD format.
    The required output format is JSON matching this structure: {StaticUserDataSchema}
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
    extracted_data = call_gemini_for_structured_output(
        system_prompt=system_prompt,
        user_text=user_text,
        user_files=user_files,
        output_schema=StaticUserData,
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
    The required output format is JSON matching this structure: {DailyUserFeelingSchema}
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
    
    extracted_data = call_gemini_for_structured_output(
        system_prompt=system_prompt,
        user_text=user_text,
        user_files=user_files,
        output_schema=DailyUserFeeling,
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
    The required output format is JSON matching this structure: {WorkoutDataSchema}
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

    extracted_data = call_gemini_for_structured_output(
        system_prompt=system_prompt,
        user_content=user_content,
        output_schema=WorkoutData,
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

def collect_weekly_summary(week_start_date: date, workouts: typing.List[WorkoutData], feelings: typing.List[DailyUserFeeling]) -> typing.Optional[WeeklyUserSummary]:
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
    The required output format is JSON matching this structure: {WeeklyUserSummarySchema}
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


    extracted_data = call_gemini_for_structured_output(
        system_prompt=system_prompt,
        user_content=full_context, # Send combined context
        user_files=user_files,
        output_schema=WeeklyUserSummary,
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

def collect_monthly_stats(year: int, month: int, user_data: StaticUserData, workouts: typing.List[WorkoutData]) -> typing.Optional[MonthlyUserStats]:
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
    The required output format is JSON matching this structure: {MonthlyUserStatsSchema}
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

    extracted_data = call_gemini_for_structured_output(
        system_prompt=system_prompt,
        user_content=full_context,
        user_files=user_files,
        output_schema=MonthlyUserStats,
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