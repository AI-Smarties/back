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

# pylint: disable=wrong-import-position,import-error
from db_utils import (
    create_category,
    create_conversation,
    create_vector,
    get_category_by_name,
    get_categories,
    get_conversations,
    get_vectors,
)
from db import create_tables

TIMEZONE = "Europe/Helsinki"


def populate_categories():
    """Create sample categories."""
    print("Creating categories...")
    categories = [
        "Eläimet",
        "Ajoneuvot",
        "Ruoka",
        "Urheilu",
        "Värit",
        "Sää",
    ]

    created_ids = {}
    for cat_name in categories:
        # Check if category already exists
        existing = get_category_by_name(cat_name)
        if existing:
            print(f"  Category '{cat_name}' already exists (ID: {existing.id})")
            created_ids[cat_name] = existing.id
        else:
            cat_id = create_category(cat_name)
            print(f"  Created category '{cat_name}' (ID: {cat_id})")
            created_ids[cat_name] = cat_id

    return created_ids


def populate_conversations(category_ids):
    """Create sample conversations."""
    print("\nCreating conversations...")
    now = datetime.now(ZoneInfo(TIMEZONE))

    conversations_data = [
        {
            "name": "Kultainennoutaja",
            "summary": "Tietoa kultaisista noutajakoirista",
            "category": "Eläimet",
            "timestamp": now - timedelta(days=5),
            "vectors": [
                "Kultaiset noutajat ovat ystävällisiä ja uskollisia koiria",
                "Niillä on kullan- tai kermanvärinen turkki",
                "Nämä koirat sopivat hyvin lapsille ja perheille",
            ],
        },
        {
            "name": "Persialainen kissa",
            "summary": "Faktoja persialaisista kissoista ja niiden hoidosta",
            "category": "Eläimet",
            "timestamp": now - timedelta(days=4),
            "vectors": [
                "Persialaisilla kissoilla on pitkä ja pörröinen turkki",
                "Ne tarvitsevat säännöllistä hoitoa ja harjausta",
                "Persialaisilla kissoilla on litteät kasvot ja pyöreät silmät",
            ],
        },
        {
            "name": "Punainen urheiluauto",
            "summary": "Keskustelua punaisista urheiluautoista",
            "category": "Ajoneuvot",
            "timestamp": now - timedelta(days=3),
            "vectors": [
                "Punainen on suosittu väri urheiluautoille",
                "Urheiluautot ovat nopeita ja niissä on tehokkaat moottorit",
                "Niissä on tyypillisesti kaksi ovea ja virtaviivainen muotoilu",
                "Punaiset urheiluautot herättävät huomiota tiellä",
            ],
        },
        {
            "name": "Toyota Yaris",
            "summary": "Tietoa Toyota Yaris -ajoneuvoista",
            "category": "Ajoneuvot",
            "timestamp": now - timedelta(days=2),
            "vectors": [
                "Toyota Yaris on rikkoutumaton auto",
                "Se tunnetaan polttoainetehokkuudestaan ja luotettavuudestaan",
                "Yaris on suosittu kaupunkiajossa",
                "Yaris on polttoainetehokas päivittäisessä työmatka-ajossa",
                "Kompakti mutta tilava suunnittelu-ihme",
            ],
        },
        {
            "name": "Pizzatäytteet",
            "summary": "Eri tyyppisiä pizzatäytteitä",
            "category": "Ruoka",
            "timestamp": now - timedelta(days=1),
            "vectors": [
                "Pepperoni on klassinen pizzatäyte",
                "Vihannekset kuten sienet ja paprikat ovat terveellisiä valintoja",
                "Juusto on pizzan tärkein ainesosa",
                "Ananas pizzassa on kiistanalainen aihe",
            ],
        },
        {
            "name": "Suklaakakku",
            "summary": "Resepti ja ideoita suklaakakkuun",
            "category": "Ruoka",
            "timestamp": now - timedelta(hours=12),
            "vectors": [
                "Suklaakakku on suosittu jälkiruoka",
                "Se vaatii jauhoja, kananmunia, kaakaojauhetta ja sokeria",
                "Kuorrute tekee suklaakakusta erityisen herkullisen",
            ],
        },
        {
            "name": "Jalkapallo-ottelu",
            "summary": "Muistiinpanoja jalkapallon säännöistä ja strategiasta",
            "category": "Urheilu",
            "timestamp": now - timedelta(hours=6),
            "vectors": [
                "Jalkapalloa pelataan 11 pelaajalla kummassakin joukkueessa",
                "Tavoitteena on tehdä maali potkaisemalla pallo verkkoon",
                "Pelaajat eivät saa käyttää käsiään paitsi maalivahti",
                "Jalkapallo-ottelut kestävät 90 minuuttia",
            ],
        },
        {
            "name": "Aurinkoinen sää",
            "summary": "Kuvaus aurinkoisista sääolosuhteista",
            "category": "Sää",
            "timestamp": now - timedelta(hours=3),
            "vectors": [
                "Aurinkoiset päivät ovat valoisia ja lämpimiä",
                "Selkeä sininen taivas ilman pilviä",
                "Hyvä sää ulkoiluun",
            ],
        },
    ]

    created_conv_ids = []
    for conv_data in conversations_data:
        cat_id = category_ids.get(conv_data["category"])
        conv_id = create_conversation(
            name=conv_data["name"],
            summary=conv_data["summary"],
            cat_id=cat_id,
            timestamp=conv_data["timestamp"],
        )
        print(f"  Created conversation '{conv_data['name']}' (ID: {conv_id})")

        # Create vectors for this conversation
        for vector_text in conv_data["vectors"]:
            vec_id = create_vector(vector_text, conv_id)
            print(f"    Created vector (ID: {vec_id})")

        created_conv_ids.append(conv_id)

    return created_conv_ids


def print_database_summary():
    """Print a summary of the database contents."""
    print("\n" + "=" * 60)
    print("DATABASE SUMMARY")
    print("=" * 60)

    categories = get_categories()
    print(f"\nTotal Categories: {len(categories)}")
    for cat in categories:
        print(f"  - {cat.name} (ID: {cat.id})")

    conversations = get_conversations()
    print(f"\nTotal Conversations: {len(conversations)}")
    for conv in conversations:
        category_name = conv.category.name if conv.category else "None"
        print(f"  - {conv.name} (ID: {conv.id}, Category: {category_name})")

    vectors = get_vectors()
    print(f"\nTotal Vectors: {len(vectors)}")

    print("\n" + "=" * 60)


def main():
    """Main function to populate the database."""
    print("Starting database population...")
    print("=" * 60)

    # Ensure tables exist
    print("\nEnsuring database tables exist...")
    create_tables()
    print("Tables ready.")

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
    main()
