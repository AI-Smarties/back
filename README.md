![GHA workflow badge](https://github.com/AI-Smarties/back/actions/workflows/main.yml/badge.svg)
[![codecov](https://codecov.io/gh/AI-Smarties/back/graph/badge.svg?token=2CMOR83HXC)](https://codecov.io/gh/AI-Smarties/back)

# AI Smarties – Backend (Python + FastAPI)

This project was created for Software Engineering Project at University of Helsinki.

[Main Repo](https://github.com/AI-Smarties/Main)

## Frontend integration

The backend is intended to be used with the [Flutter frontend](https://github.com/AI-Smarties/front)

---

## Setup and Running

### 1. Clone the repository and navigate to the directory

```bash
git clone git@github.com:AI-Smarties/back.git
```

```bash
cd back
```

### 2. Switch to development branch (dev)

```bash
git switch dev
```

### 3. Create and activate virtual environment

#### Mac / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows

```bash
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Set up PostgreSQL Database

The application uses PostgreSQL.

#### Docker Compose

```bash
docker compose up -d
```

or

```bash
docker-compose up -d
```

This starts a PostgreSQL container with the correct configuration. You can access the database at `localhost:5432`.

The application will automatically connect to `localhost` when running locally
(environment variables for the database are not required for local development).

### 6. Google Cloud Authentication (ADC)

The backend uses Google Cloud Speech-to-Text, Vertex AI API, Firebase, and the Application Default Credentials (ADC) method.

#### STEP 1: Browser Setup (Google Cloud Console)

Do these first in your web browser:

1. Create a Project: Go to [Google Cloud Console](https://console.cloud.google.com/), create a new project, and copy the Project ID (e.g., smarties-backend-v2).

2. Activate Free Trial: Click the "Activate" banner at the top of the page to claim your free credits.

3. Enable the API: Search for `"Cloud Speech-to-Text API"` in the top search bar and click Enable. Do the same for `"Vertex AI API"`

#### STEP 2: CLI Setup (Terminal)

Run these commands in your terminal:

1. Install GCloud CLI: [Download and install here](https://cloud.google.com/sdk/docs/install).

2. Log in to your Google account

    ```bash
    gcloud auth application-default login
    ```

3. Link to Project: Run this command (replace [PROJECT_ID] with the ID you copied in Step 1):

    ```bash
    gcloud auth application-default set-quota-project [PROJECT_ID]
    ```

### 7. Firebase

create the following `.env` file in the root directory:

```bash
FIREBASE_PROJECT_ID=[PROJECT_ID]
```

### 8. Start the server

```bash
fastapi run src/main.py --host 0.0.0.0
```

---

## Daily development workflow

When you return to coding, activate the virtual environment and check for updates:

1. Start database: `docker-compose up -d` (if using Docker Compose)
2. Activate environment: `source venv/bin/activate`
3. Fetch latest changes: `git pull origin dev`
4. Start server: `fastapi dev src/main.py`

---

## Project structure

- `src/main.py` – FastAPI application and WebSocket endpoint
- `src/asr.py` – Streaming ASR (Google Speech-to-Text)
- `src/db.py` – Database connection configuration
- `src/models.py` – SQLAlchemy database models
- `src/auth.py` — Firebase authentication and token verification
- `src/gemini_live.py` — Gemini Live API integration for real-time AI responses
- `src/gemini_tools.py` — Tool functions callable by Gemini during conversations
- `src/memory_extractor.py` — Extracts key data (budgets, deadlines) from transcripts and stores them to vector database
- `src/summary_service.py` — Generates meeting summaries using Gemini
- `src/context_service.py` — Builds structured context from calendar data for Gemini
- `src/db_utils.py` — Vector DB operations (store, search, retrieve embeddings)
- `requirements.txt` – Dependencies
- `docker-compose.yaml` – Local PostgreSQL setup
- `manifests/` – Deployment configurations
