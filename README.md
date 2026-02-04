![GHA workflow badge](https://github.com/AI-Smarties/back/actions/workflows/main.yml/badge.svg)
[![codecov](https://codecov.io/gh/AI-Smarties/back/graph/badge.svg?token=2CMOR83HXC)](https://codecov.io/gh/AI-Smarties/back)

# AI Smarties – Backend (Python + FastAPI)

The backend provides a WebSocket-based audio stream and real-time speech recognition using Google Speech-to-Text.

## Setup and Running

## 1. Clone repo and navigate to directory

```bash
git clone git@github.com:AI-Smarties/back.git
```
```bash
cd back
```

## 2. Switch to development branch (dev)

## Mac / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## Windows

```bash
python -m venv venv
.\venv\Scripts\activate
```

## 3. Create and activate virtual environment

```bash
python3 -m venv venv
```
```bash
source venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Google Cloud Authentication (ADC)

The backend uses Google Cloud Speech-to-Text and the Application Default Credentials (ADC) method.

### Introduction:

## STEP 1: Browser Setup (Google Cloud Console)

Do these first in your web browser:

1. Create a Project: Go to [Google Cloud Console](https://console.cloud.google.com/), create a new project, and copy the Project ID (e.g., smarties-backend-v2).

2. Activate Free Trial: Click the "Activate" banner at the top of the page to claim your free credits.

3. Enable the API: Search for "Cloud Speech-to-Text API" in the top search bar and click Enable.

## STEP 2: CLI Setup (Terminal)

Run these commands in your terminal:

1. Install GCloud CLI: [Download and install here](https://cloud.google.com/sdk/docs/install).

2. Enable the API: Search for "Cloud Speech-to-Text API" in the top search bar and click Enable.

```bash
   gcloud auth application-default login
```

3. Link to Project: Run this command (replace [PROJECT_ID] with the ID you copied in Step 1):

```bash
   gcloud auth application-default set-quota-project [PROJECT_ID]
```

## 6. Start the server

```bash
fastapi run src/main.py --host 0.0.0.0 --port 8001
```

---

## Daily development workflow

When you return to coding, activate the virtual environment and check for updates:

1. Activate environment: source venv/bin/activate
2. Fetch latest changes: git pull origin dev
3. Start server: uvicorn main:app

---

## Project structure

- `src/main.py` – FastAPI application and WebSocket endpoint
- `src/asr.py` – Streaming ASR (Google Speech-to-Text)
- `requirements.txt` – Runtime dependencies
- `venv/` – Virtual environment (not for version control)

### Frontend integration
Backend is intended to be used with the Flutter frontend.
In development, the backend runs on localhost (127.0.0.1:8001).
