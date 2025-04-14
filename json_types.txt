import sys
from datetime import date
from typing import TypedDict, Optional, List

# --- 5. Static User Data ---
class StaticUserData(TypedDict):
    """Stores baseline user information relevant for training plan personalization."""
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
