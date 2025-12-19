# CandiSight: 2-Day Hackathon Edition
## AI-Powered Candidate Screening & Resume Evaluation System

An intelligent recruitment platform that uses AI and NLP to automate resume screening and generate ATS compatibility scores.

---

## 🎯 Project Objective
Build an MVP that evaluates candidate resumes against job descriptions, predicts fit, and generates an ATS score (0-100) based on skill matching and relevance.

---

## MVP Features (2-Day Hackathon)

### 1. Resume Parsing & Analysis (CORE)
- Basic text extraction from PDF/TXT files
- Automated skill extraction
- Contact information parsing

### 2. Job Description Processing (CORE)
- Keyword/skill extraction from job description
- Simple requirement identification

### 3. Matching & Scoring Engine (CORE)
- Keyword-based skill matching
- Simple ATS compatibility score (0-100)
- Match percentage calculation

### 4. User Interface (CORE)
- Single-page resume upload
- Job description input form
- Results display with score breakdown
- Match/mismatch skills visualization

---

## Simplified Technical Architecture

```
BACKEND (Python/Flask)

├── API Endpoints
│   ├── POST /upload-resume
│   ├── POST /evaluate
│   └── GET /results/:id
├── Core Processing
│   ├── Resume Parser (PyPDF2/python-docx)
│   ├── Job Description Processor
│   └── Scoring Engine
└── In-Memory Data Store (JSON/SQLite)

FRONTEND (HTML/CSS/JavaScript)

├── Upload Form
├── Results Display
└── Score Visualization
```

---

## Technology Stack (Hackathon)

### Backend (Pick One)
**Python + Flask** (Recommended for hackathon)
  - PyPDF2 / python-docx for file processing
  - NLTK for basic NLP
  - SQLite for data storage
  - Flask-CORS for API

**Node.js + Express** (Alternative)
  - pdfparse / node-docx
  - Natural.js for NLP
  - SQLite3 or JSON file storage

### Frontend
**HTML5 + CSS3 + Vanilla JavaScript**
  - Bootstrap or Tailwind CSS for styling
  - Fetch API for backend communication
  - Chart.js for score visualization

**OR**

**React** (if team familiar)
  - Simple create-react-app
  - Axios for API calls

### Data Storage
- **SQLite** (file-based, no setup needed)
- **JSON files** (simplest option)

---

## 2-Day Hackathon Schedule

### DAY 1 (8 hours)

**Morning - Planning & Setup (3 hours: 9 AM - 12 PM)**
- [ ] Team formation and role assignment (15 min)
- [ ] Requirements finalization (30 min)
- [ ] Technology stack decision (15 min)
- [ ] Environment setup and dependencies (45 min)
- [ ] Architecture & API design (45 min)
- [ ] Break (15 min)

**Afternoon - Backend Development (5 hours: 1 PM - 6 PM)**
- [ ] Resume parser implementation (90 min)
- [ ] Job description processor (60 min)
- [ ] API endpoints setup (45 min)
- [ ] Basic scoring logic (45 min)
- [ ] Testing and debugging (15 min)

### DAY 2 (8 hours)

**Morning - Integration & Frontend (4 hours: 9 AM - 1 PM)**
- [ ] Frontend setup and structure (60 min)
- [ ] Resume upload form (45 min)
- [ ] Job description input form (30 min)
- [ ] Frontend-backend integration (45 min)
- [ ] Bug fixes and testing (20 min)

**Afternoon - Polish & Demo Prep (4 hours: 2 PM - 6 PM)**
- [ ] Results display UI (45 min)
- [ ] Score visualization (45 min)
- [ ] End-to-end testing (45 min)
- [ ] Sample data preparation (15 min)
- [ ] Demo rehearsal and documentation (30 min)
- [ ] Final testing and optimizations (20 min)

---

## Core Modules (Hackathon Version)

### 1. Resume Parser
- Extract text from PDF/TXT
- Identify skills section
- Extract candidate name, email

### 2. Job Description Processor
- Extract text
- Parse required skills
- Parse job title and level

### 3. Matching Engine
- Compare skills (case-insensitive keyword matching)
- Calculate match percentage
- Identify matched and missing skills

### 4. Scoring Engine
- Skill match score (0-100)
- Experience relevance score
- Overall ATS compatibility score
- Generate match summary

---

## Hackathon Deliverables

### Working Software
- [ ] Backend API (Flask/Express)
- [ ] Frontend web application (HTML/React)
- [ ] Resume upload and processing
- [ ] Job description input
- [ ] Scoring and results display

### Documentation
- [ ] README with setup instructions
- [ ] API endpoint documentation
- [ ] How to use guide
- [ ] Code comments

### Demo Materials
- [ ] Sample resumes and job descriptions
- [ ] Presentation slides
- [ ] Video demo (optional)

### Code Repository
- [ ] GitHub repository with clean code
- [ ] Well-organized folder structure
- [ ] .gitignore and requirements.txt/.package.json

---

## Simplified Data Models

### Resume Data
```json
{
  "id": "unique_id",
  "fileName": "resume.pdf",
  "candidateName": "John Doe",
  "email": "john@example.com",
  "extractedSkills": ["Python", "JavaScript", "React"],
  "rawText": "..."
}
```

### Job Description Data
```json
{
  "id": "job_id",
  "jobTitle": "Senior Developer",
  "description": "...",
  "requiredSkills": ["Python", "PostgreSQL", "React"]
}
```

### Evaluation Result
```json
{
  "id": "result_id",
  "resumeId": "resume_id",
  "jobId": "job_id",
  "overallScore": 78,
  "skillMatchPercentage": 75,
  "matchedSkills": ["Python", "React"],
  "missingSkills": ["PostgreSQL"],
  "recommendation": "GOOD_FIT"
}
```

---

## Hackathon Success Criteria

### Must Have (MVP)
- ✅ Functional resume upload and parsing
- ✅ Job description input
- ✅ Working matching algorithm
- ✅ Score calculation (0-100)
- ✅ Results display with matched/missing skills
- ✅ Live demo ready

### Nice to Have
- 📊 Visual score charts
- 📝 PDF report generation
- ⚡ Performance optimization
- 🎨 Polished UI design

### Hackathon Goals
- Resume processing time: < 5 seconds
- Match accuracy demonstration: ✓ Works correctly
- Clean, maintainable code
- Good presentation and demo

---

## Hackathon Challenges & Solutions

### Common Issues & Fixes

1. **Resume parsing fails on certain PDFs**
   - Solution: Use simple text extraction, test with various formats early

2. **Skill matching too strict/lenient**
   - Solution: Use case-insensitive matching, fuzzy string matching (fuzzywuzzy)

3. **Frontend-backend integration issues**
   - Solution: Start simple with JSON endpoints, use Postman to test

4. **Time running out**
   - Priority: Get basic version working, then add features
   - Skip: Database persistence (use JSON files), authentication, advanced UI

5. **Dependencies/setup issues**
   - Solution: Use simple libraries, pre-install everything, have backup tools

---

## Team Structure (Hackathon)

### Optimal Team: 3-4 People
- **Backend Developer (1-2)** - Parse, match, score, API
- **Frontend Developer (1-2)** - UI, forms, results display
- **Team Lead/Coordinator** - Planning, demos, documentation

### Flexible Assignments
- Single person: Can handle both backend and simple frontend
- 4 people: One dedicated to testing and documentation

---

## Quick Start Guide

### Before the Hackathon
1. Install Python/Node.js
2. Set up Git repository
3. Prepare sample resumes and job descriptions
4. Test file upload libraries
5. Decide on simple database (SQLite or JSON)

### First 30 Minutes
1. Clone/setup repo structure
2. Install dependencies (Flask, PyPDF2, etc.)
3. Create basic API skeleton
4. Create simple HTML form
5. Test basic file upload

### Development Priority
1. **Hour 1-3:** Resume parser + Job processor (backend)
2. **Hour 4-5:** Matching algorithm + Scoring
3. **Hour 6-8:** API endpoints + Frontend form
4. **Hour 9-14:** Frontend results display + Integration
5. **Hour 15-16:** Testing + Polish + Demo prep

---

## Final Checklist
- [ ] Code pushed to GitHub
- [ ] README with setup instructions
- [ ] At least 1 working demo example
- [ ] Presentation ready
- [ ] Team can explain every feature

---

## Project Repository Structure

```
CandiSight-Hackathon/
├── backend/
│   ├── app.py (or main.js for Node)
│   ├── resume_parser.py
│   ├── job_processor.py
│   ├── scoring_engine.py
│   ├── requirements.txt
│   └── sample_data/
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── script.js
│   └── assets/
├── data/
│   ├── resumes/
│   └── jobs/
├── .gitignore
└── README.md
```

---

## Sample API Endpoints

### 1. Upload Resume
```
POST /api/upload-resume
Content-Type: multipart/form-data

Response: 
{
  "success": true,
  "resumeId": "resume_123",
  "candidateName": "John Doe",
  "skills": ["Python", "JavaScript"]
}
```

### 2. Create Job Description
```
POST /api/create-job
Content-Type: application/json

{
  "jobTitle": "Senior Developer",
  "description": "...",
  "requiredSkills": ["Python", "React"]
}

Response:
{
  "success": true,
  "jobId": "job_456"
}
```

### 3. Evaluate Candidate
```
POST /api/evaluate
Content-Type: application/json

{
  "resumeId": "resume_123",
  "jobId": "job_456"
}

Response:
{
  "overallScore": 78,
  "skillMatchPercentage": 75,
  "matchedSkills": ["Python"],
  "missingSkills": ["React"],
  "recommendation": "GOOD_FIT"
}
```

---

## Tips for Hackathon Success

✨ **Do's:**
- Start with core features (parsing, matching, scoring)
- Use existing libraries (don't reinvent the wheel)
- Test early and often
- Keep UI simple but functional
- Document as you code
- Save frequently to Git

❌ **Don'ts:**
- Over-engineer the solution
- Try to perfect UI design
- Implement authentication
- Build for scale (one-person use is fine)
- Spend time on deployment
- Ignore error handling

---

**Good luck with your hackathon! 🚀**
