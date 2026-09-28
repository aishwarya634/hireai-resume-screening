# 🎯 HireAI: AI Resume Screening

> Screen and rank resumes against a job description in seconds, powered by LLMs.
> <img width="1261" height="627" alt="image" src="https://github.com/user-attachments/assets/18301653-2877-441d-b8cc-ecfa1820ac7e" />

<img width="1263" height="631" alt="image" src="https://github.com/user-attachments/assets/7e3b2d3c-7ea6-435b-858f-12107200e715" />
<img width="1261" height="628" alt="image" src="https://github.com/user-attachments/assets/b6261226-0a9c-45c7-9633-8068bda3d3de" />

🔗 **Live demo:** [hireai-resume-screening.vercel.app](https://hireai-resume-screening.vercel.app)

## 💡 The Problem

Recruiters and club panels spend hours reading resumes one by one. HireAI automates the first pass: it compares each resume with a job description and highlights the best matches.

## ✨ Features

- 📄 Upload resumes and paste a job description
- 🤖 AI-based analysis and matching using the Groq API
- 📊 Candidates scored and ranked by relevance
- ⚡ Optimised prompts to reduce token usage
- ☁️ Deployed on Vercel

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Backend | Python, Flask |
| AI | Groq API (LLM) |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Vercel |

## 📁 Project Structure

```
hireai-resume-screening/
├── api/            # Vercel serverless entry
├── inputs/         # Sample resumes / job descriptions
├── outputs/        # Generated results
├── app.py          # Main Flask app
├── index.html      # Frontend
└── requirements.txt
```

## 🚀 Run Locally

```bash
# 1. Clone
git clone https://github.com/aishwarya634/hireai-resume-screening.git
cd hireai-resume-screening

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
echo "GROQ_API_KEY=your_key_here" > .env

# 4. Run
python app.py
```

Then open `http://localhost:5000`.

## 🧠 What I Learned

- Building a Flask backend and deploying it as a serverless app on Vercel
- Prompt engineering and reducing LLM token usage
- Keeping API keys out of the repo with environment variables

## 🔮 Future Improvements

- [ ] Support more file formats
- [ ] Export results as CSV/PDF
- [ ] Add skill-gap suggestions for each candidate

## 👩‍💻 Author

**Aishwarya Biradar**, CS student at RNSIT
[GitHub](https://github.com/aishwarya634)
