"""Populate the database with example data via HTTP endpoints.

This script mirrors scripts/populate_db.py, but instead of importing db/db_utils
it calls the FastAPI endpoints defined in src/main.py.

By default it resets the DB by calling:
- POST /drop/tables
- POST /create/tables

Then it creates:
- categories via POST /create/category?name=...
- conversations via POST /create/conversation?name=...&summary=...&cat_id=...&timestamp=...
- vectors via POST /create/vector?text=...&conv_id=...

Usage:
  python3 scripts/populate_db_via_api.py --base-url http://localhost:8000

For staging/remote:
  python3 scripts/populate_db_via_api.py --base-url https://<host>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo
import os

from dotenv import load_dotenv


load_dotenv()


TIMEZONE = "Europe/Helsinki"


@dataclass(frozen=True)
class ApiClient:
    base_url: str
    timeout_s: float = 30.0

    def _url(self, path: str, params: dict[str, Any] | None = None) -> str:
        base = self.base_url.rstrip("/")
        if not path.startswith("/"):
            path = "/" + path
        url = base + path
        if params:
            # Keep None values out of the query string.
            clean = {k: v for k, v in params.items() if v is not None}
            if clean:
                url += "?" + urlencode(clean, doseq=True)
        return url

    token: str | None = None

    def request_json(
        self, method: str, path: str, params: dict[str, Any] | None = None
    ) -> Any:
        url = self._url(path, params=params)
        body = b"" if method.upper() in {"POST", "PUT", "PATCH", "DELETE"} else None
        req = Request(url=url, data=body, method=method.upper())
        req.add_header("Accept", "application/json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")

        try:
            with urlopen(
                req, timeout=self.timeout_s
            ) as resp:  # nosec - intended for local/staging
                raw = resp.read().decode("utf-8")
        except HTTPError as e:
            raw = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"HTTP {e.code} for {method} {url}: {raw}") from e
        except URLError as e:
            raise RuntimeError(f"Failed to reach {url}: {e}") from e

        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Non-JSON response from {url}: {raw[:2000]}") from e


def preflight_check(api: ApiClient) -> None:
    """Validate that the remote server exposes the endpoints we need.

    This prevents destructive operations (like dropping tables) on servers that
    don't actually support the full populate flow.
    """

    spec = api.request_json("GET", "/openapi.json")
    paths = set((spec or {}).get("paths", {}).keys())
    if not paths:
        raise RuntimeError("Could not read OpenAPI paths from /openapi.json")

    required = {
        "/create/category",
        "/create/conversation",
        "/create/vector",
        "/get/categories",
        "/get/conversations",
        "/get/vectors",
    }

    missing = sorted(required - paths)
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            "Server is missing required endpoints: "
            f"{missing_str}. "
            "Either deploy a backend version that includes them, "
            "or run this script against a local dev server."
        )


def reset_tables(api: ApiClient) -> None:
    print("Resetting tables via API...")
    api.request_json("POST", "/drop/tables")
    api.request_json("POST", "/create/tables")


def create_category(api: ApiClient, name: str) -> int:
    name = name.strip()
    try:
        cat = api.request_json("POST", "/create/category", params={"name": name})
        return int(cat["id"])
    except RuntimeError as e:
        # If the category already exists, fetch it.
        if "HTTP 409" not in str(e):
            raise
        cats = api.request_json("GET", "/get/categories", params={"name": name})
        if not cats:
            raise
        return int(cats[0]["id"])


def create_conversation(
    api: ApiClient,
    *,
    name: str,
    summary: str | None,
    cat_id: int | None,
    timestamp: datetime | None,
) -> int:
    params: dict[str, Any] = {"name": name.strip()}
    if summary is not None:
        params["summary"] = summary.strip()
    if cat_id is not None:
        params["cat_id"] = int(cat_id)
    if timestamp is not None:
        params["timestamp"] = timestamp.isoformat()

    conv = api.request_json("POST", "/create/conversation", params=params)
    return int(conv["id"])


def create_vector(api: ApiClient, *, text: str, conv_id: int) -> int:
    params = {"text": text.strip(), "conv_id": int(conv_id)}
    vec = api.request_json("POST", "/create/vector", params=params)
    return int(vec["id"])


def print_database_summary(api: ApiClient) -> None:
    print("\n" + "=" * 60)
    print("DATABASE SUMMARY")
    print("=" * 60)

    categories = api.request_json("GET", "/get/categories")
    conversations = api.request_json("GET", "/get/conversations")
    vectors = api.request_json("GET", "/get/vectors")

    print(f"\nTotal Categories: {len(categories or [])}")
    print(f"\nTotal Conversations: {len(conversations or [])}")
    print(f"\nTotal Vectors: {len(vectors or [])}")
    print("\n" + "=" * 60)


def populate(api: ApiClient) -> None:
    print("Creating categories...")
    category_names = [
        "Asiakaspalvelu",
        "Verkkoviat",
        "Laskutus",
        "Liittymät",
        "Laitteet",
        "Palvelupyynnöt",
    ]

    category_ids: dict[str, int] = {}
    for cat_name in category_names:
        cat_id = create_category(api, cat_name)
        print(f"  Created category '{cat_name}' (ID: {cat_id})")
        category_ids[cat_name] = cat_id

    print("\nCreating conversations...")
    now = datetime.now(ZoneInfo(TIMEZONE))

    conversations_data: list[dict[str, Any]] = [
        {
            "name": "Internet-yhteys ei toimi",
            "summary": "Asiakkaan kiinteän laajakaistan vian selvitys",
            "category": "Verkkoviat",
            "timestamp": now - timedelta(days=5),
            "vectors": [
                "Asiakas ilmoittaa ettei internet-yhteys toimi lainkaan",
                "Modeemin uudelleenkäynnistys ei ratkaissut ongelmaa",
                "Verkossa havaittu alueellinen häiriö",
            ],
        },
        {
            "name": "Hidas mobiilidata",
            "summary": "Mobiiliverkon nopeusongelman käsittely",
            "category": "Verkkoviat",
            "timestamp": now - timedelta(days=4),
            "vectors": [
                "Asiakas kokee mobiilidatan olevan erittäin hidasta",
                "Verkon kuormitus ruuhka-aikana vaikuttaa nopeuteen",
                "Suositellaan verkon tilan tarkistusta ja tukiaseman vaihtoa",
            ],
        },
        {
            "name": "Laskun epäselvyys",
            "summary": "Asiakas kysyy lisämaksuista laskulla",
            "category": "Laskutus",
            "timestamp": now - timedelta(days=3),
            "vectors": [
                "Asiakas huomasi ylimääräisiä maksuja laskulla",
                "Lisämaksu johtuu palvelupaketin muutoksesta",
                "Selitetty laskun erittely asiakkaalle",
            ],
        },
        {
            "name": "Liittymän päivitys 5G:hen",
            "summary": "Asiakas haluaa päivittää liittymänsä",
            "category": "Liittymät",
            "timestamp": now - timedelta(days=2),
            "vectors": [
                "Asiakas haluaa siirtyä 5G-liittymään",
                "Tarkistettu laitteen yhteensopivuus",
                "Liittymän päivitys onnistuu välittömästi",
            ],
        },
        {
            "name": "Modeemin asennus",
            "summary": "Ohjeistus uuden modeemin käyttöönottoon",
            "category": "Laitteet",
            "timestamp": now - timedelta(days=1),
            "vectors": [
                "Asiakas tarvitsee ohjeet modeemin asennukseen",
                "Modeemi kytketään sähköön ja verkkoon",
                "Yhteys tarkistetaan laitteen hallintasivulta",
            ],
        },
        {
            "name": "SIM-kortti ei toimi",
            "summary": "SIM-kortin aktivointiongelma",
            "category": "Liittymät",
            "timestamp": now - timedelta(hours=12),
            "vectors": [
                "Asiakas ei saa yhteyttä uudella SIM-kortilla",
                "SIM-kortti ei ole aktivoitunut järjestelmässä",
                "Aktivointi tehty ja yhteys palautui",
            ],
        },
        {
            "name": "Palvelupyyntö: muutto",
            "summary": "Liittymän siirto uuteen osoitteeseen",
            "category": "Palvelupyynnöt",
            "timestamp": now - timedelta(hours=6),
            "vectors": [
                "Asiakas ilmoittaa muutosta uuteen osoitteeseen",
                "Palvelun saatavuus tarkistettu uudessa kohteessa",
                "Muuttopäivälle tehty palvelunsiirto",
            ],
        },
        {
            "name": "Yleinen asiakaspalvelukysely",
            "summary": "Neuvontaa palveluiden käytöstä",
            "category": "Asiakaspalvelu",
            "timestamp": now - timedelta(hours=3),
            "vectors": [
                "Asiakas kysyy palveluiden käytöstä",
                "Annettu ohjeet ja lisätietoa palveluista",
                "Ohjattu tarvittaessa jatkotukeen",
            ],
        },
    ]

    for conv_data in conversations_data:
        cat_id = category_ids.get(conv_data["category"])
        conv_id = create_conversation(
            api,
            name=conv_data["name"],
            summary=conv_data["summary"],
            cat_id=cat_id,
            timestamp=conv_data["timestamp"],
        )
        print(f"  Created conversation '{conv_data['name']}' (ID: {conv_id})")

        for vector_text in conv_data["vectors"]:
            information = f"Konteksti: {conv_data['name']}; Sisältö: {vector_text}"
            vec_id = create_vector(api, text=information, conv_id=conv_id)
            print(f"    Created vector (ID: {vec_id})")
            import time as t

            t.sleep(13)
            #  Add time.sleep() here if facing problems with too fast population


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate DB via FastAPI HTTP endpoints"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL for the backend (e.g. http://localhost:8000 or https://example.com)",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not drop/create tables before populating",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Firebase ID token for Authorization header",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token or os.environ.get("TOKEN")
    api = ApiClient(base_url=args.base_url, timeout_s=args.timeout, token=token)

    print("Starting database population via API...")
    print("=" * 60)
    print(f"Base URL: {api.base_url}")

    preflight_check(api)

    if not args.no_reset:
        reset_tables(api)

    populate(api)
    print_database_summary(api)
    print("\n✓ Database population completed successfully!")


if __name__ == "__main__":
    main()
