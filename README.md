![GHA workflow badge](https://github.com/AI-Smarties/back/actions/workflows/main.yml/badge.svg)

# AI Smarties – Backend (Python + Django)

## Asennus ja ajaminen

## 1. Kloonaa repo ja mene hakemistoon

```bash
git clone git@github.com:AI-Smarties/back.git
```
```bash
cd back
```

## 2. Vaihda kehityshaaraan (dev)

```bash
git checkout dev
```

## 3. Luo ja aktivoi virtuaaliympäristö

```bash
python3 -m venv venv
```
```bash
source venv/bin/activate
```

## 4. Asenna riippuvuudet ja aja migraatiot

```bash
pip install -r requirements.txt
```
```bash
python manage.py migrate
```

## 5. Käynnistä palvelin

```bash
python manage.py runserver
```

---

## Päivittäinen kehitystyö

Kun palaat koodaamaan, aktivoi virtuaaliympäristö ja tarkista päivitykset:

1. Aktivoi ympäristö: source venv/bin/activate
2. Hae uusimmat muutokset: git pull origin dev
3. Käynnistä palvelin: python manage.py runserver

---

## Projektin rakenne

- api/ – Django-sovellus (API-päätepisteet ja logiikka)
- config/ – Projektin asetukset ja konfiguraatio
- manage.py – Djangon hallintatyökalu
- requirements.txt – Python-riippuvuudet
