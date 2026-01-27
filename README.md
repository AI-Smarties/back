![GHA workflow badge](https://github.com/AI-Smarties/back/actions/workflows/main.yml/badge.svg)
[![codecov](https://codecov.io/gh/AI-Smarties/back/graph/badge.svg?token=2CMOR83HXC)](https://codecov.io/gh/AI-Smarties/back)

# AI Smarties – Backend (Python + Django)

## Setup and Running

## 1. Clone repo and navigate to directory

```bash
git clone git@github.com:AI-Smarties/back.git
```
```bash
cd back
```

## 2. Switch to development branch (dev)

```bash
git checkout dev
```

## 3. Create and activate virtual environment

```bash
python3 -m venv venv
```
```bash
source venv/bin/activate
```

## 4. Install dependencies and run migrations

```bash
pip install -r requirements.txt
```
```bash
python manage.py migrate
```

## 5. Start the server

```bash
python manage.py runserver
```

---

## Daily development workflow

When you return to coding, activate the virtual environment and check for updates:

1. Activate environment: source venv/bin/activate
2. Fetch latest changes: git pull origin dev
3. Start server: python manage.py runserver

---

## Project structure

- api/ – Django-sovellus (API-päätepisteet ja logiikka)
- config/ – Projektin asetukset ja konfiguraatio
- manage.py – Djangon hallintatyökalu
- requirements.txt – Python-riippuvuudet

---

## API

### POST /api/message/

Request body:
{
  "text": "string"
}

Response:
{
  "reply": "string"
}

---

### Frontend integration
Backend is intended to be used with the Flutter frontend.
In development, the backend runs on localhost (127.0.0.1:8000).
