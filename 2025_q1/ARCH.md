# Application Architecture and Flow

## Detailed Application Flow - From Launch to Completion

### **Phase 1: Application Startup**

When you run `python main.py` (or `python main.py --verbose` for debugging mode), the following initialization sequence occurs:

1. **Environment Detection & Configuration**
   - Checks if running in Kaggle container or locally
   - Sets up data directories (`logs/` for storage, `data/` for input files)
   - Configures Google API key (from Kaggle Secrets or environment variable)
   - Sets data retention periods (30/60/180/365 days for different data types)

2. **Data File Initialization**
   - Creates empty JSON arrays for: daily_feelings, workout_data, weekly_summaries, monthly_stats
   - Does NOT create user_data.json (created during first-time setup)

### **Phase 2: Automatic Data Import**

The app checks for CSV files in `new_data/` directory:

1. **Workout Import** (`workouts.csv`)
   - Parses date/time, duration, distance, workout type
   - Calculates pace if not provided
   - Adds default values for pain/exertion (5/0/0/100)
   - Deduplicates by date+time
   - Merges with existing data, sorts by date
   - Deletes CSV after successful import

2. **Daily Metrics Import** (`daily_metrics.csv`)
   - Maps HRV to stress levels (lower HRV = higher stress)
   - Maps resting HR to overall feeling (lower HR = better feeling)
   - Adds defaults for energy/shin_pain/sleep (3/0/3)
   - Deduplicates by date
   - Deletes CSV after import

### **Phase 3: Automatic Data Purge**

Removes old data based on retention settings:
- Daily feelings older than 30 days
- Weekly summaries older than 180 days
- Workouts older than 60 days
- Monthly stats older than 365 days

### **Phase 4: Interactive Data Collection**

1. **User Profile** (first time only)
   - Multimodal prompt: "Tell me about yourself"
   - Accepts text, audio, images
   - Extracts: name, age, height, gender, fitness goal, target date, preferred long run day
   - Uses structured output with retry logic

2. **Daily Feeling** (if missing for today)
   - AI conversationally asks about:
     - Overall feeling (1-5)
     - Energy level (1-5)
     - Shin pain (0-5)
     - Sleep quality (1-5)
   - Optional: stress, hydration, nutrition, notes

3. **Workout Logging** (user-driven loop)
   - Asks "Any workouts to log today?"
   - For each workout:
     - Type, duration, exertion (1-10)
     - Pain levels: shin/knee (0-5), tightness (0-5)
     - Adherence percentage
     - Optional: distance, pace, HR, elevation
   - Supports workout app screenshots

4. **Weekly Summary** (if last week not summarized)
   - Calculates stats from logged workouts
   - User provides:
     - Hardest/easiest workout descriptions
     - Week completion status
     - Adherence percentage
     - General notes

5. **Monthly Statistics** (if last month not summarized)
   - Calculates totals from workout data
   - User provides:
     - Injury days count
     - Weight/HR changes
     - General observations

### **Phase 5: Workout Plan Generation**

1. **Data Aggregation**
   - Combines all user data into single context
   - Includes current date/day for proper scheduling

2. **AI Plan Creation**
   - Generates structured plan with:
     - Assessment of last workout
     - Progress evaluation
     - 7-day detailed schedule
     - Next workout specifics
     - Alternative workout option
     - Motivational message

3. **Review Loop**
   - User can: accept, request changes, or start over
   - Changes are applied iteratively
   - Saves final plan to `current_workout_plan.json`

### **Phase 6: Interactive Q&A**

1. **Chat Session**
   - New AI conversation with plan context
   - User asks questions about the plan
   - AI provides explanations, alternatives, tips
   - Offers to end session every 3 questions
   - Exit keywords: 'exit', 'quit', 'done', 'bye'

### **Key Technical Details**

**Structured Output Generation**:
- Two-phase approach: AI generates, then validates/fixes
- Up to 10 retry attempts with smart backoff
- Tracks missing fields across attempts
- Falls back to manual entry after 3 consecutive failures

**Data Persistence**:
- Custom JSON encoder/decoder for datetime handling
- Immediate saves after each collection
- Graceful handling of missing/corrupt files

**Error Handling**:
- API rate limits: uses retry_delay from response
- Service errors: exponential backoff
- Invalid JSON: regeneration with enhanced prompts
- File operations: creates directories as needed

The entire flow ensures data completeness at each step, with no progression until required information is collected. The modular design allows easy modification of individual phases without affecting others.

## Architecture Overview

### Directory Structure
```
src/
├── __init__.py          # Package initialization
├── config.py            # Configuration and environment setup
├── data_models.py       # TypedDict definitions and JSON schemas
├── utils.py             # Utility functions (JSON, date handling)
├── api_client.py        # Gemini API integration
├── data_collection.py   # Data collection functions
├── data_import.py       # CSV data import functionality
├── data_purge.py        # Auto-purge old data functionality
└── workout_planning.py  # Plan generation and Q&A
main.py                  # Application entry point
sample_data_loader.py    # Sample data for testing
```

### Core Components

#### Config Module (`src/config.py`)
- Handles environment detection (Kaggle vs local)
- Manages API key configuration
- Defines file paths for data storage
- Configures data retention periods (can be overridden via environment variables):
  - `DATA_RETENTION_DAILY_FEELINGS` (default: 30 days)
  - `DATA_RETENTION_WORKOUT_DATA` (default: 60 days)
  - `DATA_RETENTION_WEEKLY_SUMMARIES` (default: 180 days)
  - `DATA_RETENTION_MONTHLY_STATS` (default: 365 days)

#### Data Models (`src/data_models.py`)
- Contains all TypedDict class definitions
- Defines JSON schemas for structured output validation
- Central location for data structure documentation

#### API Client (`src/api_client.py`)
- `upload_file_to_gemini()`: Handles file uploads
- `get_user_input_with_multimodal()`: Supports text, image, and audio input
- `call_gemini_for_structured_output()`: Implements two-phase structured generation

#### Data Collection (`src/data_collection.py`)
- Individual collection functions for each data type
- Each function handles prompting, collection, and validation
- Saves data to appropriate JSON files

#### Workout Planning (`src/workout_planning.py`)
- `plan_workout()`: Generates personalized workout plans
- `workout_q_and_a()`: Interactive Q&A session
- Handles plan modifications and user feedback
- Provides current date and day of week to the model for proper scheduling

#### Data Import (`src/data_import.py`)
- `import_csv_data()`: Main function that imports CSV files from `new_data/`
- `parse_workout_csv()`: Parses workouts.csv with deduplication
- `parse_daily_metrics_csv()`: Parses daily_metrics.csv with deduplication
- Maps external data formats to internal data models
- Automatically deletes CSV files after successful import

#### Data Purge (`src/data_purge.py`)
- `purge_all_old_data()`: Main function that purges all data types
- `purge_daily_feelings()`: Removes daily feelings older than retention period
- `purge_workout_data()`: Removes workouts older than retention period
- `purge_weekly_summaries()`: Removes weekly summaries older than retention period
- `purge_monthly_stats()`: Removes monthly stats older than retention period
- Retention periods are configurable via environment variables

### Data Storage

All data is persisted in JSON format in the `logs/` directory:
- `logs/user_data.json`: User profile
- `logs/daily_feelings.json`: Daily subjective feelings
- `logs/workout_data.json`: Workout history
- `logs/weekly_summaries.json`: Weekly summaries
- `logs/monthly_stats.json`: Monthly statistics
- `logs/current_workout_plan.json`: Latest generated plan

### Google Gemini Integration

Uses Google's Gemini AI API with:
- Model: `gemini-2.5-flash-exp`
- Structured output generation for data collection
- Interactive information gathering with conversation loops
- Multimodal support (text, images, audio)
- Context caching for efficient processing
- Advanced retry logic:
  - Smart retry using API-provided wait times from retry_delay field
  - Automatic retry for service errors (503)
  - Re-generation for invalid JSON responses
  - Missing field tracking across attempts
  - Manual intervention after 3 consecutive failures for the same fields
  - Up to 10 total attempts before giving up
  - Total failure fallback: Direct manual collection of all fields
  - Enhanced prompts for better field extraction
  - Smart type conversion and validation for manual inputs

The API key should be set as an environment variable: `GOOGLE_API_KEY`

## Terminal Features

The application includes readline support for better terminal interaction:
- Use **Up/Down arrow keys** to navigate through previous inputs
- Command history is maintained within the session
- Works automatically on platforms with readline support (most Unix/Linux/macOS systems)

## Verbose Mode

The application supports a verbose debugging mode that provides detailed information about:

1. **Command Usage**
   - Run with `python main.py --verbose` or `python main.py -v`
   - Enables detailed logging throughout the application

2. **What Verbose Mode Shows**
   - System prompts sent to the AI model
   - User inputs and combined prompts
   - Full conversation history during structured output generation
   - Raw JSON responses from the model (especially useful for debugging malformed JSON)
   - Model names and configuration details
   - Token/character counts for prompts and responses
   - Detailed error messages with stack traces
   - API retry attempts and wait times

3. **Key Debug Points**
   - **Structured Output Generation**: Shows both info-gathering and JSON generation phases
   - **JSON Parsing Errors**: Displays the full malformed JSON response and exact error position
   - **Data Collection**: Shows raw user input and how it's processed
   - **Workout Planning**: Displays system instructions and full prompts

4. **Use Cases**
   - Debugging JSON decode errors from the LLM
   - Understanding why certain fields are missing in structured output
   - Monitoring API interactions and retry behavior
   - Optimizing prompts based on actual model responses