"""
Script to populate the database with example data (without vectors).

This script creates sample categories and conversations for testing purposes,
but skips creating vectors to avoid requiring embedding model credentials.
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
    get_category_by_name,
    get_categories,
    get_conversations,
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
        },
        {
            "name": "Persialainen kissa",
            "summary": "Faktoja persialaisista kissoista ja niiden hoidosta",
            "category": "Eläimet",
            "timestamp": now - timedelta(days=4),
        },
        {
            "name": "Punainen urheiluauto",
            "summary": "Keskustelua punaisista urheiluautoista",
            "category": "Ajoneuvot",
            "timestamp": now - timedelta(days=3),
        },
        {
            "name": "Toyota Yaris",
            "summary": "Tietoa Toyota Yaris -ajoneuvoista",
            "category": "Ajoneuvot",
            "timestamp": now - timedelta(days=2),
        },
        {
            "name": "Pizzatäytteet",
            "summary": "Eri tyyppisiä pizzatäytteitä",
            "category": "Ruoka",
            "timestamp": now - timedelta(days=1),
        },
        {
            "name": "Suklaakakku",
            "summary": "Resepti ja ideoita suklaakakkuun",
            "category": "Ruoka",
            "timestamp": now - timedelta(hours=12),
        },
        {
            "name": "Jalkapallo-ottelu",
            "summary": "Muistiinpanoja jalkapallon säännöistä ja strategiasta",
            "category": "Urheilu",
            "timestamp": now - timedelta(hours=6),
        },
        {
            "name": "Aurinkoinen sää",
            "summary": "Kuvaus aurinkoisista sääolosuhteista",
            "category": "Sää",
            "timestamp": now - timedelta(hours=3),
        },
        {
            "name": "Vihreä väri",
            "summary": "Tietoa vihreästä väristä",
            "category": "Värit",
            "timestamp": now - timedelta(hours=1),
        },
        {
            "name": "Sateinen päivä",
            "summary": "Muistiinpanoja sateisista sääilmiöistä",
            "category": "Sää",
            "timestamp": now - timedelta(minutes=30),
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

    print("\n" + "=" * 60)


def main():
    """Main function to populate the database."""
    print("Starting database population (without vectors)...")
    print("=" * 60)

    # Ensure tables exist
    print("\nEnsuring database tables exist...")
    create_tables()
    print("Tables ready.")

    try:
        # Create categories
        category_ids = populate_categories()

        # Create conversations
        populate_conversations(category_ids)

        # Print summary
        print_database_summary()

        print("\n✓ Database population completed successfully!")
        print("\nNote: This script creates categories and conversations only.")
        print("Use populate_db.py to also create vectors (requires credentials).")

    except Exception as e:
        print(f"\n✗ Error during database population: {e}")
        raise


if __name__ == "__main__":
    main()
