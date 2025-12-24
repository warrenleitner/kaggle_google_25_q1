# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GenAI fitness coaching application that uses Google's Gemini AI to generate personalized workout plans. The app demonstrates structured output generation, multimodal input (images, audio), and function calling.

## Commands

- To run the main application: `python ai_fitness_coach.py`
- To test API functionality: `python test_genai_api.py`
- Install dependencies: `pip install google-generativeai` (Note: requirements.txt is not present yet)
- Set up API key: `export GOOGLE_API_KEY="your-key-here"` (Linux/Mac) or `set GOOGLE_API_KEY=your-key-here` (Windows)

## Architecture

The application is currently transitioning from a monolithic structure to a modular architecture:

### Current State
- **ai_fitness_coach.py**: Main application (Jupyter notebook converted to Python)
- **test_genai_api.py**: API exploration and testing
- **logs/**: JSON files storing user data and workout plans
- **data/**: Sample input files (images, audio, CSV)
- **new_data/**: Directory for importing new CSV data

### Planned Modular Structure (per ARCH.md)
```
src/
├── config.py            # Configuration and environment setup
├── data_models.py       # TypedDict definitions and JSON schemas
├── utils.py             # Utility functions
├── api_client.py        # Gemini API integration
├── data_collection.py   # Data collection functions
├── data_import.py       # CSV data import functionality
├── data_purge.py        # Auto-purge old data functionality
└── workout_planning.py  # Plan generation and Q&A
main.py                  # Application entry point (not yet created)
```

## Key Functions

### Core GenAI Functions
- `call_gemini_for_structured_output()`: Two-phase approach for structured JSON generation
- `get_user_input_with_multimodal()`: Accepts text, images, and audio input

### Data Collection Flow
1. Static user profile (one-time)
2. Daily feelings
3. Workout logs
4. Weekly summaries
5. Monthly statistics

### Workflow Phases
1. **Data Import**: Automatically imports CSV files from `new_data/`
2. **Data Purge**: Removes old data based on retention periods
3. **Data Collection**: Interactive gathering of user data
4. **Plan Generation**: Creates personalized workout plans
5. **Q&A Session**: Interactive questions about the plan

## Google Gemini Integration

- Model: `gemini-2.5-flash-exp`
- Advanced retry logic with smart backoff
- Supports multimodal input and structured output
- Function calling for interactive data gathering

## Development Guidelines

### Current Focus Areas
- The codebase is in transition - `ai_fitness_coach.py` contains the working implementation
- The modular structure described in ARCH.md is the target architecture
- Missing files that need creation: `main.py`, `requirements.txt`, entire `src/` directory

### When Making Changes
- Update ARCH.md for architectural changes
- Update README.md for user-facing features
- Follow the existing retry and error handling patterns
- Maintain JSON data persistence in `logs/` directory

### Data Retention Periods
- Daily feelings: 30 days
- Workout data: 60 days  
- Weekly summaries: 180 days
- Monthly stats: 365 days

These can be overridden via environment variables (e.g., `DATA_RETENTION_DAILY_FEELINGS`)