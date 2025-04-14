import sys
from datetime import date
from typing import TypedDict, Optional, List

# --- 5. Static User Data ---
class StaticUserData(TypedDict):
    """Stores baseline user information relevant for training plan personalization."""
    name: str  # User's name
    age: int  # User's age in years
    height_inches: Optional[float] # User's height in inches (optional)
    gender: Optional[str] # User's reported gender (optional, e.g., "Male", "Female", "Non-binary", "Prefer not to say")
     # --- Goal --- 
    goal: str  # User's primary goal (e.g., "5K", "10K", "Half Marathon", "Marathon", "Ultra", "Other")
    goal_time: Optional[str]  # Target time for the goal (e.g., 2.5 for 2 hours 30 minutes)
    goal_date: date  # Date of the user would like to be ready for their goal. (e.g., "2024-05-15")
     # --- Training Preferences (Optional) ---
    preferred_training_days: Optional[List[str]] # Days user prefers to train (e.g., ["Monday", "Wednesday", "Friday", "Sunday"])
    long_run_day: Optional[str] # Preferred day for the weekly long run (e.g., "Saturday")
     # --- Other Optional Data ---
    notes: Optional[str]  # Optional free-text notes for additional context

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
       "format": "date"
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
class DailyUserFeeling(TypedDict):
    """Subjective user input regarding their physical and mental state on a specific day."""
    date: date  # The date this feeling log applies to
    overall_feeling: int  # Rating scale, e.g., 1 (Very Poor) to 5 (Excellent)
    energy_level: int  # Rating scale, e.g., 1 (Very Low) to 5 (Very High)
    shin_pain: int  # Rating scale, e.g., 0 (None) to 5 (Severe)
    sleep_quality: int  # Rating scale, e.g., 1 (Very Poor) to 5 (Excellent)
    stress_level: Optional[int]  # Optional: 1 (Very Low) to 5 (Very High)
    hydration_level: Optional[int]  # 1 (Poor) to 5 (Excellent). Note that this represents the previous day
    nutrition_quality: Optional[int]  # 1 (Poor) to 5 (Excellent). Note that this represents the previous day
    notes: Optional[str]  # Optional free-text notes

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
      "format": "date"
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
class WorkoutData(TypedDict):
    """Data for a single workout session (running or other types)."""
    date: date  # Date of the workout
    workout_type: str  # E.g., "Easy Run", "Tempo", "Strength", "Yoga", "Cycling", "Other"
    perceived_exertion: int  # Optional: RPE scale 1-10
    shin_and_knee_pain: int  # Optional: 0 (None) to 5 (Severe) (Specific to this user's shin pain)
    shin_tightness: int  # Optional: 0 (None) to 5 (Severe) (Specific to this user's shin tightness)
    workout_adherence: int # Percent adherence to the planned workout (0-100%)
    actual_duration_minutes: float # Total workout time in minutes
    actual_distance_miles: Optional[float]  # Distance in miles (None if not applicable)
    average_pace_minutes_per_mile: Optional[float]  # Pace in min/mile (None if not applicable)
    average_heart_rate_bpm: Optional[int]
    max_heart_rate_bpm: Optional[int]
    elevation_gain_feet: Optional[float]  # Elevation gain in feet (None if not applicable)
    notes: Optional[str]  # User comments, could include exercises/sets/reps for strength

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
      "format": "date"
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
class WeeklyUserSummary(TypedDict):
    """A qualitative summary of the user's training week."""
    week_start_date: date  # E.g., the Monday of the week
    overall_summary: str  # Text summary of the week (consistency, feeling, progress)
    key_achievements: Optional[List[str]]  # Bullet points of successes
    areas_for_focus: Optional[List[str]]  # Bullet points for improvement areas
    total_workouts: int # Total number of workouts logged this week
    # Optional quantitative context if available/relevant
    total_running_distance_miles: Optional[float]
    total_workout_duration_minutes: Optional[float]

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
      "format": "date"
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
class MonthlyUserStats(TypedDict):
    """Aggregated user statistics for a specific month."""
    month: str  # Format: "YYYY-MM"
    # --- Optional Physiological Estimates ---
    weight_pounds: float # User's current weight in pounds
    max_heart_rate_bpm: Optional[int] # Max HR (optional, estimated or tested)
    resting_heart_rate_bpm: Optional[int] # Resting HR (optional)
    vo2_max_estimated: Optional[float] # Estimated VO2 Max (optional)
    longest_run_distance_miles: float
    average_pace_minutes_per_mile: Optional[float] # Avg pace for runs this month
    comfortable_pace_minutes_per_mile: Optional[float] # Comfortable pace for runs this month
    comfortable_run_distance_miles: Optional[float] # Avg comfortable run distance
    average_heart_rate_bpm: Optional[int] # Avg HR across workouts with HR data
    total_elevation_gain_feet: Optional[float] # Total elevation for runs
    monthly_summary_notes: Optional[str] # Optional field for brief LLM-generated or user notes

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