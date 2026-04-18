"""
Script to populate the database with example data.

This script creates sample categories, conversations, and vectors
for testing and demonstration purposes.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Add the src directory to the path so we can import our modules
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# pylint: disable=duplicate-code,wrong-import-position,import-error
from db_utils import (
    create_category,
    create_conversation,
    create_vector,
    get_categories,
    get_conversations,
    get_vectors,
)
from db import create_tables, drop_tables

TIMEZONE = "Europe/Helsinki"
POPULATE_USER_ID = "populate-script-user"  # Put your own user ID here


def populate_categories():
    """Create sample categories."""
    print("Creating categories...")
    categories = [
        "Asiakaspalvelu",
        "Verkkoviat",
        "Laskutus",
        "Liittymät",
        "Laitteet",
        "Palvelupyynnöt",
    ]

    created_ids = {}
    for cat_name in categories:
        cat = create_category(cat_name, user_id=POPULATE_USER_ID)
        print(f"  Created category '{cat_name}' (ID: {cat.id})")
        created_ids[cat_name] = cat.id

    return created_ids


def populate_conversations(category_ids):
    """Create sample conversations."""
    print("\nCreating conversations...")
    now = datetime.now(ZoneInfo(TIMEZONE))

    conversations_data = [
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

    created_conv_ids = []
    for conv_data in conversations_data:
        cat_id = category_ids.get(conv_data["category"])
        conv = create_conversation(
            name=conv_data["name"],
            summary=conv_data["summary"],
            cat_id=cat_id,
            timestamp=conv_data["timestamp"],
            user_id=POPULATE_USER_ID,
        )
        print(f"  Created conversation '{conv_data['name']}' (ID: {conv.id})")

        # Create vectors for this conversation
        for vector_text in conv_data["vectors"]:
            information = f"Konteksti: {conv_data['name']}; Sisältö: {vector_text}"
            vec = create_vector(information, conv.id)
            print(f"    Created vector (ID: {vec.id})")

        created_conv_ids.append(conv.id)

    return created_conv_ids


def print_database_summary():
    """Print a summary of the database contents."""
    print("\n" + "=" * 60)
    print("DATABASE SUMMARY")
    print("=" * 60)

    categories = get_categories(POPULATE_USER_ID)
    print(f"\nTotal Categories: {len(categories)}")

    conversations = get_conversations(POPULATE_USER_ID)
    print(f"\nTotal Conversations: {len(conversations)}")

    vectors = get_vectors(POPULATE_USER_ID)
    print(f"\nTotal Vectors: {len(vectors)}")

    print("\n" + "=" * 60)


def main():
    """Main function to populate the database."""
    print("Starting database population...")
    print("=" * 60)

    try:
        # Create categories
        category_ids = populate_categories()

        # Create conversations and vectors
        populate_conversations(category_ids)

        # Print summary
        print_database_summary()

        print("\n✓ Database population completed successfully!")

    except Exception as e:
        print(f"\n✗ Error during database population: {e}")
        raise


if __name__ == "__main__":
    drop_tables()
    create_tables()
    main()
