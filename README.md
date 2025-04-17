# GenAI Workout Coach

An open-source, AI-powered fitness coaching assistant. GenAI Workout Coach leverages Google's Gemini LLM and multimodal input to deliver personalized, holistic workout planning—bridging the gap between expensive, rigid fitness apps and truly adaptive coaching.

---

## 🚀 Features

- **Personalized Workout Plans:**  
  AI-generated plans adapt to your goals, preferred activities, and progress.

- **Multimodal Data Input:**  
  Accepts workout information via text, screenshots (images), and audio recordings for rich, flexible data entry.

- **Structured Output (JSON):**  
  All user and plan data is stored and handled in flexible JSON—making progress tracking robust and extensible.

- **Feedback & Conversational Adjustments:**  
  Easily provide feedback and guidance; the AI coach updates your plan in real-time.

- **Holistic Approach:**  
  Supports running, strength training, and general wellness—tailoring guidance across disciplines.

- **Q&A Mode:**  
  Ask about exercise form, clarify terms, or seek general advice without editing your plan.

---

## 🗂️ Example Dataset

Want to see a sample coaching workflow?  
Check out the [GenAI Coaching Dataset on Kaggle](http://kaggle.com/warrenleitner/kaggle-google-25-q1-data).

---

## 🛠️ Tech Stack

- **Python** (>=3.9)
- **Google Gemini API**
- **JSON** for data persistence & structure
- Local file storage for seamless offline use

---

## ⚡ Usage

1. **Clone the Repository**
```git clone https://github.com/yourusername/genai-workout-coach.git cd genai-workout-coach```


2. **Install Requirements**
```pip install -r requirements.txt```

3. **Set Up API Key**  
Obtain your **Google Gemini API Key** and set it as an environment variable:
```export GOOGLE_API_KEY="your_google_gemini_api_key"```

4. **Run the Application**
```python ai_fitness_coach_md.py```

The app guides you through initial profile setup, multimodal data input, and interactive plan generation.

---

## 🏗️ How It Works

- **Data Gathering**
- Collect user stats, goals, and workout data via text, image, or audio.
- **Plan Generation**
- The AI processes your aggregated data and provides a week-by-week workout plan in markdown.
- **Refinement & Feedback**
- Accept the plan or submit suggestions for further adaptation.
- **Q&A Session**
- Ask any questions about your regimen or fitness concepts—get context-aware responses.
- **Persistence**
- Data (user profile, logs, plans) are saved in JSON for next time.

---

## 🔮 Roadmap

- [ ] **Apple Health** integration for auto-importing biometric & workout data
- [ ] **Dynamic schemas** and import/export support (e.g., .fit format)
- [ ] **Progress tracking and analytics** dashboards
- [ ] **Improved Q&A** with broader exercise science knowledge

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

**Ready to upgrade your digital fitness journey?  
Try the [sample on Kaggle](http://kaggle.com/warrenleitner/kaggle-google-25-q1-data) or get started now!**
