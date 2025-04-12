"""
Prototype AI Fitness Coach CLI Application for Kaggle.

This script implements a basic command-line AI fitness coach using Langchain
and Google Gemini. It manages user data, prompts for updates, generates
workout plans, and answers questions.

Requirements:
- langchain
- langchain-google-genai
- google-generativeai
- python-dotenv (optional, for local testing - use Kaggle secrets in production)

Setup in Kaggle:
1. Add your Google API Key to Kaggle Secrets. Key name: GOOGLE_API_KEY
2. Ensure the required libraries are installed in your Kaggle notebook environment:
   !pip install langchain langchain-google-genai google-generativeai python-dotenv --upgrade --quiet
"""

import json
import os
import sys
from datetime import date, datetime, timedelta
from typing import TypedDict, Optional, List, Type, Dict, Any, Union
import warnings

# --- Suppress specific warnings ---
# Suppress UserWarnings from google.generativeai related to retries
warnings.filterwarnings("ignore", category=UserWarning, module='google.generativeai.client')
# Suppress ResourceWarnings (can occur in async environments like notebooks)
warnings.filterwarnings("ignore", category=ResourceWarning)

# --- Attempt to import necessary libraries ---
try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain.chains import LLMChain
    from langchain.memory import ConversationBufferMemory
    # Try to load .env for local dev, ignore if it fails or running in Kaggle
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("ERROR: Required libraries not found.")
    print("Please install them: pip install langchain langchain-google-genai google-generativeai python-dotenv --upgrade")
    sys.exit(1)

# --- Configuration ---
# Retrieve API Key from Kaggle secrets or environment variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("ERROR: Google API Key not found.")
    print("Please add it to Kaggle Secrets (Key: GOOGLE_API_KEY) or set the environment variable.")
    # In a real Kaggle notebook, you might exit or raise an error.
    # For this script, we'll allow it to proceed but LLM calls will fail.
    # sys.exit(1)
    print("WARNING: Proceeding without API key. LLM calls will fail.")


LLM_MODEL_NAME = "gemini-1.5-flash-latest"
DATA_DIR = "/kaggle/working/" # Standard Kaggle writable directory

# --- File Paths ---
STATIC_USER_DATA_FILE = os.path.join(DATA_DIR, "static_user_data.json")
DAILY_FEELINGS_FILE = os.path.join(DATA_DIR, "daily_feelings.json")
WORKOUT_LOGS_FILE = os.path.join(DATA_DIR, "workout_logs.json")
WEEKLY_SUMMARIES_FILE = os.path.join(DATA_DIR, "weekly_summaries.json")
MONTHLY_STATS_FILE = os.path.join(DATA_DIR, "monthly_stats.json")
WORKOUT_PLAN_FILE = os.path.join(DATA_DIR, "current_workout_plan.json") # To store the accepted plan

# --- Data Type Definitions (Copied from json_types.py for a single script) ---

# Use BaseModel for Langchain structured output
class StaticUserData(BaseModel):
    """Stores baseline user information relevant for training plan personalization."""
    age: int = Field(description="User's age in years")
    height_inches: Optional[float] = Field(None, description="User's height in inches")
    gender: Optional[str] = Field(None, description="User's reported gender (e.g., 'Male', 'Female', 'Non-binary', 'Prefer not to say')")
    goal: str = Field(description="User's primary goal (e.g., '5K', '10K', 'Half Marathon', 'Marathon', 'Ultra', 'Other')")
    goal_time: Optional[str] = Field(None, description="Target time for the goal (e.g., 'Sub 2 hours', '1:55:00', 'Around 45 minutes')")
    goal_date: str = Field(description="Date user wants to be ready for their goal (YYYY-MM-DD)")
    preferred_training_days: Optional[List[str]] = Field(None, description="Days user prefers to train (e.g., ['Monday', 'Wednesday', 'Friday', 'Sunday'])")
    long_run_day: Optional[str] = Field(None, description="Preferred day for the weekly long run (e.g., 'Saturday')")
    notes: Optional[str] = Field(None, description="Optional free-text notes for additional context")

class DailyUserFeeling(BaseModel):
    """Subjective user input regarding their physical and mental state on a specific day."""
    date: str = Field(description="The date this feeling log applies to (YYYY-MM-DD)")
    overall_feeling: int = Field(description="Rating scale: 1 (Very Poor) to 5 (Excellent)")
    energy_level: int = Field(description="Rating scale: 1 (Very Low) to 5 (Very High)")
    shin_pain: int = Field(description="Rating scale: 0 (None) to 5 (Severe)")
    sleep_quality: int = Field(description="Rating scale: 1 (Very Poor) to 5 (Excellent)")
    stress_level: Optional[int] = Field(None, description="Optional: Rating scale: 1 (Very Low) to 5 (Very High)")
    hydration_level: Optional[int] = Field(None, description="Rating scale for *previous* day: 1 (Poor) to 5 (Excellent)")
    nutrition_quality: Optional[int] = Field(None, description="Rating scale for *previous* day: 1 (Poor) to 5 (Excellent)")
    notes: Optional[str] = Field(None, description="Optional free-text notes")

class WorkoutData(BaseModel):
    """Data for a single workout session (running or other types)."""
    date: str = Field(description="Date of the workout (YYYY-MM-DD)")
    workout_type: str = Field(description="E.g., 'Easy Run', 'Tempo', 'Intervals', 'Long Run', 'Strength', 'Yoga', 'Cycling', 'Rest', 'Other'")
    perceived_exertion: Optional[int] = Field(None, description="Optional: RPE scale 1 (Very Easy) to 10 (Max Effort)")
    shin_and_knee_pain: Optional[int] = Field(None, description="Optional: Pain during/after workout: 0 (None) to 5 (Severe)")
    shin_tightness: Optional[int] = Field(None, description="Optional: Tightness during/after workout: 0 (None) to 5 (Severe)")
    workout_adherence: Optional[int] = Field(None, description="Optional: How well the planned workout was followed (0-100%)")
    actual_duration_minutes: Optional[float] = Field(None, description="Total workout time in minutes")
    actual_distance_miles: Optional[float] = Field(None, description="Distance in miles (if applicable)")
    average_pace_minutes_per_mile: Optional[float] = Field(None, description="Pace in min/mile (if applicable)")
    average_heart_rate_bpm: Optional[int] = Field(None, description="Average HR (if tracked)")
    max_heart_rate_bpm: Optional[int] = Field(None, description="Max HR (if tracked)")
    elevation_gain_feet: Optional[float] = Field(None, description="Elevation gain in feet (if applicable)")
    notes: Optional[str] = Field(None, description="User comments, could include exercises/sets/reps for strength, details about the run, etc.")

class WeeklyUserSummary(BaseModel):
    """A qualitative summary of the user's training week."""
    week_start_date: str = Field(description="The start date (usually Monday) of the week being summarized (YYYY-MM-DD)")
    overall_summary: str = Field(description="Text summary of the week (consistency, feeling, progress, challenges)")
    key_achievements: Optional[List[str]] = Field(None, description="Bullet points of successes or highlights")
    areas_for_focus: Optional[List[str]] = Field(None, description="Bullet points for improvement areas or things to watch")
    # These counts can be derived, but prompting helps get user's perspective
    total_workouts_this_week: Optional[int] = Field(None, description="User's estimate of total workouts logged this week")
    total_running_distance_miles_this_week: Optional[float] = Field(None, description="User's estimate of total running distance this week")

class MonthlyUserStats(BaseModel):
    """Aggregated user statistics and feelings for a specific month."""
    month: str = Field(description="Month being summarized (Format: 'YYYY-MM')")
    weight_pounds: Optional[float] = Field(None, description="User's approximate weight in pounds at the end of the month")
    max_heart_rate_bpm: Optional[int] = Field(None, description="Updated Max HR (if known/tested)")
    resting_heart_rate_bpm: Optional[int] = Field(None, description="Updated Resting HR (if known)")
    vo2_max_estimated: Optional[float] = Field(None, description="Updated Estimated VO2 Max (if known)")
    longest_run_distance_miles_this_month: Optional[float] = Field(None, description="Longest single run distance this month")
    average_pace_minutes_per_mile_this_month: Optional[float] = Field(None, description="Approximate average pace for runs this month")
    comfortable_pace_minutes_per_mile: Optional[float] = Field(None, description="Approximate comfortable/easy pace currently")
    comfortable_run_distance_miles: Optional[float] = Field(None, description="Typical distance for a comfortable/easy run")
    monthly_summary_notes: Optional[str] = Field(None, description="Overall thoughts, feelings, progress, or challenges for the month")


# --- Utility Functions ---

def safe_load_json(filepath: str, default: Any = None) -> Any:
    """Safely loads JSON data from a file."""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"Data file not found: {filepath}. Returning default.")
            return default
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {filepath}. Returning default.")
        return default
    except Exception as e:
        print(f"An unexpected error occurred loading {filepath}: {e}. Returning default.")
        return default

def save_json(filepath: str, data: Any):
    """Saves data to a JSON file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, default=str) # Use default=str for dates
        print(f"Data saved to {filepath}")
    except Exception as e:
        print(f"Error saving data to {filepath}: {e}")

def get_last_date(data_list: List[Dict[str, Any]], date_key: str = "date") -> Optional[date]:
    """Gets the latest date from a list of dictionaries."""
    if not data_list:
        return None
    try:
        latest_date = max(datetime.strptime(item[date_key], "%Y-%m-%d").date() for item in data_list if date_key in item)
        return latest_date
    except (ValueError, KeyError, TypeError):
        print(f"Warning: Could not parse dates using key '{date_key}' in data list.")
        return None

def get_start_of_week(dt: date) -> date:
    """Returns the Monday of the week for the given date."""
    return dt - timedelta(days=dt.weekday())

def get_start_of_month(dt: date) -> str:
    """Returns the YYYY-MM string for the given date."""
    return dt.strftime("%Y-%m")

def print_separator():
    """Prints a visual separator to the console."""
    print("\n" + "="*60 + "\n")

# --- LLM Interaction ---

def initialize_llm(api_key: Optional[str]) -> Optional[ChatGoogleGenerativeAI]:
    """Initializes the Langchain ChatGoogleGenerativeAI model."""
    if not api_key:
        print("LLM initialization skipped: API key is missing.")
        return None
    try:
        genai.configure(api_key=api_key)
        # Adjust temperature for more deterministic JSON output when needed
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0.2, convert_system_message_to_human=True)
        print(f"Initialized LLM: {LLM_MODEL_NAME}")
        return llm
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return None

def get_structured_llm_output(
    llm: ChatGoogleGenerativeAI,
    system_prompt: str,
    data_schema: Type[BaseModel],
    user_input_prompt: str = "Please provide the information. You can describe feelings, workout details, upload files (by describing their content if actual upload isn't possible here), etc. Type 'done' when finished, even if some fields are missing."
) -> Optional[Dict[str, Any]]:
    """
    Interacts with the LLM to fill a Pydantic schema based on user input.

    Args:
        llm: Initialized Langchain ChatGoogleGenerativeAI model.
        system_prompt: The instruction/context for the LLM.
        data_schema: The Pydantic BaseModel class defining the desired output structure.
        user_input_prompt: The initial prompt displayed to the user.

    Returns:
        A dictionary matching the schema or None if unable to get valid data.
    """
    if not llm:
        print("LLM is not available.")
        return None

    try:
        # Use structured output feature (akin to function calling)
        structured_llm = llm.with_structured_output(data_schema)

        print_separator()
        print(f"AI Coach: {system_prompt}")
        print_separator()

        collected_input = ""
        while True:
            print(f"AI Coach: {user_input_prompt}")
            user_response = input("You: ")
            if user_response.strip().lower() == 'done':
                if not collected_input:
                    print("AI Coach: No information provided. Skipping this data entry.")
                    return None
                print("AI Coach: Okay, attempting to process the information provided...")
                break
            collected_input += user_response + "\n"
            user_input_prompt = "Provide more details or type 'done'." # Follow-up prompt

        try:
            # Combine system prompt (implicitly handled by `with_structured_output` context)
            # and the collected user input for the final LLM call.
            # We pass the system prompt again here just to be explicit in the thought process,
            # though `with_structured_output` primarily uses the schema.
            # A more robust approach might use a ChatPromptTemplate.
            print("AI Coach: Processing your input with the LLM...")
            ai_msg = structured_llm.invoke(f"{system_prompt}\n\nPlease extract the information from the following user input:\n{collected_input}")

            if isinstance(ai_msg, data_schema):
                print("AI Coach: Successfully extracted information.")
                return ai_msg.dict()
            else:
                print(f"AI Coach: Unexpected response type from LLM: {type(ai_msg)}. Failed to extract structured data.")
                # Maybe try a fallback or ask user to rephrase? For prototype, we fail here.
                return None

        except Exception as e:
            print(f"AI Coach: Error during LLM structured output processing: {e}")
            print("AI Coach: Could not automatically extract all information. Please ensure you provided clear details.")
            return None # Indicate failure

    except Exception as e:
        print(f"An unexpected error occurred during LLM interaction: {e}")
        return None

# --- Data Collection Prompts ---

def get_static_user_data_prompt() -> str:
    return (
        "Welcome to the AI Fitness Coach setup!\n"
        "To personalize your training, I need some baseline information. "
        "Please tell me about yourself, your fitness goals, and preferences. "
        "Key things I need are your age, primary goal (like '5K run', 'Marathon', 'General Fitness'), "
        "your target date or timeframe for this goal, and optionally, things like your height, gender, "
        "preferred training days, and a target time if you have one."
    )

def get_daily_feeling_prompt(today_str: str) -> str:
    return (
        f"Good [morning/afternoon/evening]! Let's log how you're feeling for today, {today_str}.\n"
        "Please rate the following on a scale (e.g., 1-5 where 5 is best, unless noted otherwise):\n"
        "- Overall feeling (1=Very Poor, 5=Excellent)\n"
        "- Energy level (1=Very Low, 5=Very High)\n"
        "- Shin pain (0=None, 5=Severe) - *Answer 0 if no shin issues today!*\n"
        "- Sleep quality last night (1=Very Poor, 5=Excellent)\n"
        "- Optional: Stress level (1=Very Low, 5=Very High)\n"
        "- Optional: Hydration yesterday (1=Poor, 5=Excellent)\n"
        "- Optional: Nutrition yesterday (1=Poor, 5=Excellent)\n"
        "You can also add any free-text notes."
    )

def get_workout_data_prompt(today_str: str) -> str:
    return (
        f"Time to log a workout! Please describe a recent session (ideally from today, {today_str}, or yesterday).\n"
        "Include:\n"
        "- Date of the workout (YYYY-MM-DD)\n"
        "- Workout type (e.g., Easy Run, Tempo, Strength, Yoga, Rest)\n"
        "- Duration (minutes)\n"
        "- Optional: Distance (miles), Average Pace (min/mile), Average/Max HR, Elevation Gain (feet)\n"
        "- Optional: Perceived Exertion (RPE 1-10)\n"
        "- Optional: Shin/Knee Pain during/after (0-5)\n"
        "- Optional: Shin Tightness during/after (0-5)\n"
        "- Optional: How well you adhered to any planned workout (0-100%)\n"
        "- Optional: Notes (e.g., exercises done, how the run felt, weather)."
    )

def get_weekly_summary_prompt(week_start_str: str) -> str:
    return (
        f"Let's summarize your training week starting {week_start_str}.\n"
        "Reflect on the past 7 days:\n"
        "- Provide an overall summary: How did the week go? Consistency? Energy levels? Progress? Challenges?\n"
        "- List any key achievements or highlights.\n"
        "- Note any areas you want to focus on or things to improve next week.\n"
        "- Optionally, estimate the total number of workouts and total running distance for the week."
    )

def get_monthly_stats_prompt(month_str: str) -> str:
    return (
        f"Let's capture some stats and reflections for the month of {month_str}.\n"
        "Please provide updates on any of the following if you know them:\n"
        "- Approximate weight (pounds)\n"
        "- Max Heart Rate (if tested/updated)\n"
        "- Resting Heart Rate (if measured)\n"
        "- Estimated VO2 Max (if known)\n"
        "- Longest run distance this month (miles)\n"
        "- Approximate average running pace this month (min/mile)\n"
        "- Current comfortable/easy run pace (min/mile)\n"
        "- Typical distance for a comfortable/easy run (miles)\n"
        "- Add any overall summary notes, thoughts, or feelings about your training this past month."
    )


# --- Main Application Logic ---

def run_fitness_coach():
    """Runs the main loop of the AI Fitness Coach CLI application."""

    llm = initialize_llm(GOOGLE_API_KEY)
    if not llm:
        print("Exiting application due to LLM initialization failure.")
        # In a real app, might offer to proceed without LLM features or retry.
        # For this prototype, we stop if LLM isn't ready.
        return

    today = date.today()
    today_str = today.strftime("%Y-%m-%d")
    current_week_start = get_start_of_week(today)
    current_week_start_str = current_week_start.strftime("%Y-%m-%d")
    current_month_str = get_start_of_month(today)

    print("--- AI Fitness Coach Initializing ---")

    # 1. Load Data
    static_data = safe_load_json(STATIC_USER_DATA_FILE)
    daily_feelings = safe_load_json(DAILY_FEELINGS_FILE, [])
    workout_logs = safe_load_json(WORKOUT_LOGS_FILE, [])
    weekly_summaries = safe_load_json(WEEKLY_SUMMARIES_FILE, [])
    monthly_stats = safe_load_json(MONTHLY_STATS_FILE, [])

    # 2. Initial Setup / Data Update Prompts
    if not static_data:
        print("Performing one-time user setup...")
        system_prompt = get_static_user_data_prompt()
        extracted_data = get_structured_llm_output(llm, system_prompt, StaticUserData)
        if extracted_data:
            # Validate/Convert date format if needed (LLM might return different formats)
            try:
                datetime.strptime(extracted_data.get("goal_date", ""), "%Y-%m-%d")
            except (ValueError, TypeError):
                 print("AI Coach: Invalid or missing goal date format from LLM. Requesting again or setting default.")
                 # In a real app, you'd loop here or handle the error more gracefully
                 extracted_data["goal_date"] = today_str # Placeholder
            static_data = extracted_data
            save_json(STATIC_USER_DATA_FILE, static_data)
        else:
            print("Failed to collect initial user data. Exiting.")
            return # Cannot proceed without basic info


    # Check and prompt for updates sequentially
    last_feeling_date = get_last_date(daily_feelings, "date")
    if not last_feeling_date or last_feeling_date < today:
        print_separator()
        print(f"Daily check-in needed for {today_str}.")
        system_prompt = get_daily_feeling_prompt(today_str)
        extracted_data = get_structured_llm_output(llm, system_prompt, DailyUserFeeling,
                                                   user_input_prompt=f"How are you feeling today ({today_str})? Describe your ratings or type 'done'.")
        if extracted_data:
             # Ensure date is correctly set
             extracted_data["date"] = today_str
             daily_feelings.append(extracted_data)
             save_json(DAILY_FEELINGS_FILE, daily_feelings)

    last_workout_date = get_last_date(workout_logs, "date")
    # Simple check: prompt if no workout logged for today or yesterday
    if not last_workout_date or last_workout_date < (today - timedelta(days=1)):
         print_separator()
         print(f"Workout log needed (haven't seen one for today or yesterday).")
         system_prompt = get_workout_data_prompt(today_str)
         extracted_data = get_structured_llm_output(llm, system_prompt, WorkoutData,
                                                    user_input_prompt=f"Log a workout (ideally from {today_str} or yesterday):")
         if extracted_data:
              # Validate date if needed, though prompt asks for it
              try:
                  datetime.strptime(extracted_data.get("date", ""), "%Y-%m-%d")
              except (ValueError, TypeError):
                  print("AI Coach: Invalid or missing workout date. Setting to today.")
                  extracted_data["date"] = today_str # Default if missing/invalid
              workout_logs.append(extracted_data)
              save_json(WORKOUT_LOGS_FILE, workout_logs)


    last_weekly_date = get_last_date(weekly_summaries, "week_start_date")
    if not last_weekly_date or last_weekly_date < current_week_start:
         print_separator()
         print(f"Weekly summary needed for week starting {current_week_start_str}.")
         system_prompt = get_weekly_summary_prompt(current_week_start_str)
         extracted_data = get_structured_llm_output(llm, system_prompt, WeeklyUserSummary,
                                                    user_input_prompt=f"Summarize your training week starting {current_week_start_str}:")
         if extracted_data:
              extracted_data["week_start_date"] = current_week_start_str
              weekly_summaries.append(extracted_data)
              save_json(WEEKLY_SUMMARIES_FILE, weekly_summaries)

    last_monthly_date_str = None
    if monthly_stats:
        try:
            last_monthly_date_str = max(item["month"] for item in monthly_stats if "month" in item)
        except (ValueError, KeyError, TypeError):
            print("Warning: Could not determine last monthly summary date.")

    if not last_monthly_date_str or last_monthly_date_str < current_month_str:
         print_separator()
         print(f"Monthly stats needed for {current_month_str}.")
         system_prompt = get_monthly_stats_prompt(current_month_str)
         extracted_data = get_structured_llm_output(llm, system_prompt, MonthlyUserStats,
                                                    user_input_prompt=f"Provide stats/reflections for {current_month_str}:")
         if extracted_data:
              extracted_data["month"] = current_month_str
              monthly_stats.append(extracted_data)
              save_json(MONTHLY_STATS_FILE, monthly_stats)


    print_separator()
    print("--- All Data Up-to-Date ---")

    # Combine all data for context
    full_context = {
        "static_user_data": static_data,
        "daily_feelings": daily_feelings,
        "workout_logs": workout_logs,
        "weekly_summaries": weekly_summaries,
        "monthly_stats": monthly_stats
    }
    context_json_str = json.dumps(full_context, indent=2, default=str)

    # 3. Workout Planning Phase
    print_separator()
    print("--- Workout Planning ---")

    planning_system_prompt = f"""
You are an AI Fitness Coach. Your task is to generate a personalized workout plan (e.g., for the next day or week) based on the user's static data, recent feelings, workout history, weekly summaries, and monthly stats.

Analyze the provided data carefully, paying attention to:
- User's goal ({static_data.get('goal', 'N/A')}), goal date ({static_data.get('goal_date', 'N/A')}), and target time ({static_data.get('goal_time', 'N/A')}).
- Preferred training days and long run day.
- Recent feelings: energy levels, shin pain (!), sleep quality.
- Recent workouts: type, duration, distance, pace, perceived exertion, reported pain/tightness. Adherence to previous plans.
- Weekly summaries: overall feeling, achievements, focus areas.
- Monthly stats: progress markers, comfortable pace/distance.

**Your Goal:** Create a balanced and progressive workout suggestion. Consider rest days. Be mindful of reported pain (especially shin pain - adjust intensity or suggest alternatives/rest if necessary). Explain the *reasoning* behind your suggestions. Ask the user for feedback.

**User Data:**
```json
{context_json_str}
```

Start by proposing a plan (e.g., for tomorrow or the next few days). Then, engage in a conversation to refine it based on user feedback.
"""

    # Using ConversationBufferMemory for planning interaction
    planning_memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    planning_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(planning_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )
    planning_chain = LLMChain(
        llm=llm, # Use the standard LLM here for conversation
        prompt=planning_prompt,
        verbose=False, # Set to True for debugging
        memory=planning_memory
    )

    print("AI Coach: Based on your data, I'll propose a workout plan. Let's discuss it.")
    # Initial prompt to kick off planning (could be empty or a specific starter)
    plan_response = planning_chain.predict(input="Let's start planning.")
    print(f"AI Coach:\n{plan_response}")

    accepted_plan = None
    while True:
        user_feedback = input("You (feedback/questions, or type 'accept'): ")
        if user_feedback.strip().lower() == 'accept':
            print("AI Coach: Great! Plan accepted.")
            # Save the last AI response as the plan
            accepted_plan = plan_response
            save_json(WORKOUT_PLAN_FILE, {"accepted_plan_text": accepted_plan, "accepted_date": today_str})
            break
        elif user_feedback.strip().lower() in ['quit', 'exit']:
            print("AI Coach: Okay, ending planning session.")
            break

        plan_response = planning_chain.predict(input=user_feedback)
        print(f"AI Coach:\n{plan_response}")

    # 4. Q&A Phase (only if a plan was accepted)
    if accepted_plan:
        print_separator()
        print("--- Q&A Session ---")

        qa_system_prompt = f"""
You are an AI Fitness Coach assistant. The user has just accepted the following workout plan. Your role now is to answer any questions they have about the plan, their training data, general fitness advice, or related topics.

Use the user's historical data and the accepted plan as context for your answers.

**Accepted Workout Plan:**
```
{accepted_plan}
```

**User Data:**
```json
{context_json_str}
```

Be helpful, encouraging, and informative. If you don't know an answer, say so.
"""
        # Use a new memory buffer for Q&A
        qa_memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        qa_prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(qa_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{input}")
            ]
        )
        qa_chain = LLMChain(
            llm=llm,
            prompt=qa_prompt,
            verbose=False,
            memory=qa_memory
        )

        print("AI Coach: Do you have any questions about the plan or anything else? (Type 'done' when finished)")
        while True:
            user_question = input("You (question or type 'done'): ")
            if user_question.strip().lower() == 'done':
                print("AI Coach: Okay, ending Q&A session.")
                break
            elif user_question.strip().lower() in ['quit', 'exit']:
                 print("AI Coach: Okay, ending Q&A session.")
                 break

            ai_answer = qa_chain.predict(input=user_question)
            print(f"AI Coach:\n{ai_answer}")

    print_separator()
    print("--- AI Fitness Coach Session Complete ---")

# --- Entry Point ---
if __name__ == "__main__":
    # Add a check for the API key again right before running
    if not GOOGLE_API_KEY:
        print("ERROR: Cannot run the coach without a Google API Key.")
        print("Please configure Kaggle Secrets or environment variables.")
    else:
        run_fitness_coach()
