"""
Integration tests for evaluate_db_data (Flash decision-making).

These tests hit the real Gemini Flash API — they require valid
Google Application Default Credentials (ADC) to run.

Run with:
    pytest src/tests/evaluate_db_data_test.py -v
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import gemini_tools
from gemini_tools import evaluate_db_data


@pytest.fixture(autouse=True)
def reset_gemini_client():
    """Reset the global genai CLIENT between tests to avoid event loop conflicts."""
    gemini_tools.CLIENT = None
    yield
    gemini_tools.CLIENT = None


def make_vector(text: str, timestamp_str: str = "2026-04-01 10:00:00+00:00") -> object:
    """Create a minimal mock Vector with the fields evaluate_db_data uses."""
    ts = datetime.fromisoformat(timestamp_str)
    conversation = SimpleNamespace(timestamp=ts)
    return SimpleNamespace(text=text, conversation=conversation)


# ---------------------------------------------------------------------------
# CASE 1: User already stated the information — Flash must NOT return it
# ---------------------------------------------------------------------------

class TestUserAlreadyKnows:

    @pytest.mark.asyncio
    async def test_persian_cat_user_states_all_facts(self):
        """
        User describes Persian cat characteristics themselves.
        Vectors contain the same facts.
        Flash must return not_relevant.
        """
        transcript = (
            "Minulla on persialainen kissa kotona. "
            "Niillä on litteät kasvot ja pitkä pörröinen turkki. "
            "Pitää muistaa harjata niitä säännöllisesti."
        )
        vectors = [
            make_vector("Persialaisilla kissoilla on litteät kasvot ja pyöreät silmät"),
            make_vector("Persialaisilla kissoilla on pitkä ja pörröinen turkki"),
            make_vector("Ne tarvitsevat säännöllistä hoitoa ja harjausta"),
        ]
        result = await evaluate_db_data(
            transcript=transcript,
            vector_database_response=vectors,
            thinking_context="User is discussing Persian cat characteristics and care.",
            query_history=[],
        )
        assert result["status"] == "not_relevant", (
            f"Flash should NOT return info user already stated. Got: {result}"
        )

    @pytest.mark.asyncio
    async def test_sunny_weather_user_describes_it(self):
        """
        User describes the weather themselves.
        Vectors confirm the same weather facts.
        Flash must return not_relevant.
        """
        transcript = (
            "Tänään on tosi aurinkoinen päivä. "
            "Taivas on täysin selkeä ja sininen. "
            "Hyvä sää ulkoiluun."
        )
        vectors = [
            make_vector("Aurinkoiset päivät ovat valoisia ja lämpimiä"),
            make_vector("Selkeä sininen taivas ilman pilviä"),
            make_vector("Hyvä sää ulkoiluun"),
        ]
        result = await evaluate_db_data(
            transcript=transcript,
            vector_database_response=vectors,
            thinking_context="User commented on the clear sunny weather.",
            query_history=[],
        )
        assert result["status"] == "not_relevant", (
            f"Flash should NOT return info user already stated. Got: {result}"
        )


# ---------------------------------------------------------------------------
# CASE 2: User asks for info they do NOT have — Flash must return it
# ---------------------------------------------------------------------------

class TestUserNeedsInfo:

    @pytest.mark.asyncio
    async def test_football_user_asks_for_more(self):
        """
        User knows 11 players and goalkeeper rule but asks for more.
        Vectors contain match duration (90 min) and scoring rule — new info.
        Flash must return found with the new info.
        """
        transcript = (
            "Jalkapalloa pelataan 11 vastaan 11 ja maalivahti saa käyttää käsiä. "
            "Onko sulla muuta tietoa jalkapallosta?"
        )
        vectors = [
            make_vector("Jalkapallo-ottelut kestävät 90 minuuttia"),
            make_vector("Tavoitteena on tehdä maali potkaisemalla pallo verkkoon"),
            make_vector("Jalkapalloa pelataan 11 pelaajalla kummassakin joukkueessa"),
            make_vector("Pelaajat eivät saa käyttää käsiään paitsi maalivahti"),
        ]
        result = await evaluate_db_data(
            transcript=transcript,
            vector_database_response=vectors,
            thinking_context="User explicitly asked for more information about football.",
            query_history=[],
        )
        assert result["status"] == "found", (
            f"Flash should return new football info user asked for. Got: {result}"
        )
        assert "90" in result["information"], (
            f"Response should mention match duration (90 min). Got: {result['information']}"
        )

    @pytest.mark.asyncio
    async def test_chocolate_cake_ingredients_unknown(self):
        """
        User wants to make chocolate cake but doesn't know ingredients.
        Vectors contain the ingredients.
        Flash must return found with ingredients.
        """
        transcript = "Haluaisin tehdä suklaakakun mutta en tiedä mitä aineksia tarvitaan."
        vectors = [
            make_vector("Suklaakakku vaatii jauhoja, kananmunia, kaakaojauhetta ja sokeria"),
            make_vector("Kuorrute tekee suklaakakusta erityisen herkullisen"),
            make_vector("Suklaakakku on suosittu jälkiruoka"),
        ]
        result = await evaluate_db_data(
            transcript=transcript,
            vector_database_response=vectors,
            thinking_context="User asked for chocolate cake ingredients.",
            query_history=[],
        )
        assert result["status"] == "found", (
            f"Flash should return cake ingredients. Got: {result}"
        )
        assert any(word in result["information"].lower() for word in ["jauhoja", "kananmunia", "kaakao", "sokeria"]), (
            f"Response should contain ingredients. Got: {result['information']}"
        )

    @pytest.mark.asyncio
    async def test_pizza_toppings_user_asks_for_suggestions(self):
        """
        User asks for additional pizza topping ideas.
        Vectors contain topping suggestions not mentioned in transcript.
        Flash must return found with new suggestions.
        """
        transcript = "Tilasin pizzan. Mitä muita täytteitä pizzaan voi laittaa?"
        vectors = [
            make_vector("Vihannekset kuten sienet ja paprikat ovat terveellisiä valintoja"),
            make_vector("Ananas pizzassa on kiistanalainen aihe"),
            make_vector("Juusto on pizzan tärkein ainesosa"),
        ]
        result = await evaluate_db_data(
            transcript=transcript,
            vector_database_response=vectors,
            thinking_context="User is asking for pizza topping suggestions.",
            query_history=[],
        )
        assert result["status"] == "found", (
            f"Flash should return topping suggestions. Got: {result}"
        )


# ---------------------------------------------------------------------------
# CASE 3: Response language must match vectors (Finnish)
# ---------------------------------------------------------------------------

class TestResponseLanguage:

    @pytest.mark.asyncio
    async def test_response_in_finnish_when_vectors_in_finnish(self):
        """
        Vectors are in Finnish.
        Flash must respond in Finnish, not English.
        """
        transcript = "Kerro Toyota Yariksen ominaisuuksista."
        vectors = [
            make_vector("Toyota Yaris tunnetaan polttoainetehokkuudestaan ja luotettavuudestaan"),
            make_vector("Yaris on suosittu kaupunkiajossa"),
        ]
        result = await evaluate_db_data(
            transcript=transcript,
            vector_database_response=vectors,
            thinking_context="User asked about Toyota Yaris features.",
            query_history=[],
        )
        assert result["status"] == "found", f"Should find Yaris info. Got: {result}"
        # Check response is Finnish — look for Finnish characters or common Finnish words
        info = result["information"].lower()
        assert any(word in info for word in ["yaris", "polttoaine", "kaupunki", "luotettav", "suosittu"]), (
            f"Response should be in Finnish. Got: {result['information']}"
        )
        assert "known for" not in info and "popular" not in info, (
            f"Response should not be in English. Got: {result['information']}"
        )


# ---------------------------------------------------------------------------
# CASE 4: Do not repeat already sent info (query_history)
# ---------------------------------------------------------------------------

class TestQueryHistory:

    @pytest.mark.asyncio
    async def test_does_not_repeat_already_sent_info(self):
        """
        Flash already sent Toyota Yaris info in this session.
        Same vectors come again.
        Flash must return not_relevant.
        """
        transcript = "Puhutaan autoista. Toyota Yarisista on hyvä polttoainetehokkuus."
        vectors = [
            make_vector("Toyota Yaris tunnetaan polttoainetehokkuudestaan ja luotettavuudestaan"),
            make_vector("Yaris on suosittu kaupunkiajossa"),
        ]
        query_history = [
            {
                "query": "Toyota Yaris features",
                "thinking_context": "User discussed Toyota Yaris.",
                "answer": "Toyota Yaris tunnetaan polttoainetehokkuudestaan ja luotettavuudestaan.",
            }
        ]
        result = await evaluate_db_data(
            transcript=transcript,
            vector_database_response=vectors,
            thinking_context="User is still discussing Toyota Yaris.",
            query_history=query_history,
        )
        assert result["status"] == "not_relevant", (
            f"Flash should not repeat already sent info. Got: {result}"
        )


# ---------------------------------------------------------------------------
# CASE 5: Completely unrelated vectors — Flash must return not_relevant
# ---------------------------------------------------------------------------

class TestUnrelatedVectors:

    @pytest.mark.asyncio
    async def test_unrelated_vectors_not_returned(self):
        """
        User talks about football.
        Vectors contain only cat information.
        Flash must return not_relevant.
        """
        transcript = "Jalkapallossa pelataan 11 vastaan 11."
        vectors = [
            make_vector("Persialaisilla kissoilla on litteät kasvot ja pyöreät silmät"),
            make_vector("Kultaiset noutajat ovat ystävällisiä koiria"),
        ]
        result = await evaluate_db_data(
            transcript=transcript,
            vector_database_response=vectors,
            thinking_context="User is discussing football rules.",
            query_history=[],
        )
        assert result["status"] == "not_relevant", (
            f"Flash should not return unrelated cat info for football query. Got: {result}"
        )


# ---------------------------------------------------------------------------
# CASE 6: Partial overlap — user mentions part of vector content
# ---------------------------------------------------------------------------

class TestPartialOverlap:

    @pytest.mark.asyncio
    async def test_user_mentions_part_vector_has_more(self):
        """
        User says "litteät kasvot".
        Vector says "litteät kasvot ja pyöreät silmät".
        Flash should NOT return it — the extra info ("pyöreät silmät") is
        trivially linked to what user already mentioned and adds no value.
        This is the critical partial overlap bug.
        """
        transcript = "Persialaisilla kissoilla on litteät kasvot."
        vectors = [
            make_vector("Persialaisilla kissoilla on litteät kasvot ja pyöreät silmät"),
        ]
        result = await evaluate_db_data(
            transcript=transcript,
            vector_database_response=vectors,
            thinking_context="User mentioned Persian cat flat faces.",
            query_history=[],
        )
        assert result["status"] == "not_relevant", (
            f"Flash should NOT return near-duplicate info. Got: {result}"
        )

    @pytest.mark.asyncio
    async def test_user_mentions_fur_vector_adds_grooming_detail(self):
        """
        User says "kissoilla pitkä turkki".
        Vector says "pitkä turkki vaatii päivittäistä hoitoa".
        The grooming detail is minor extension of what user already said.
        Flash should return not_relevant.
        """
        transcript = "Persialaisilla kissoilla on pitkä turkki."
        vectors = [
            make_vector("Persialaisilla kissoilla on pitkä turkki joka vaatii päivittäistä harjausta"),
        ]
        result = await evaluate_db_data(
            transcript=transcript,
            vector_database_response=vectors,
            thinking_context="User mentioned Persian cat long fur.",
            query_history=[],
        )
        assert result["status"] == "not_relevant", (
            f"Flash should NOT return minor extension of already stated info. Got: {result}"
        )


# ---------------------------------------------------------------------------
# CASE 7: Multi-vector combining — only new parts should be returned
# ---------------------------------------------------------------------------

class TestMultiVectorCombining:

    @pytest.mark.asyncio
    async def test_combined_vectors_only_new_part_returned(self):
        """
        User already knows: "jalkapalloa pelataan 11 pelaajalla".
        Vectors contain: 11 players (known) + 90 minutes (new) + scoring rule (new).
        Flash must return ONLY the new info, not the already-known part.
        """
        transcript = "Jalkapalloa pelataan 11 pelaajalla kummassakin joukkueessa."
        vectors = [
            make_vector("Jalkapalloa pelataan 11 pelaajalla kummassakin joukkueessa"),
            make_vector("Jalkapallo-ottelut kestävät 90 minuuttia"),
            make_vector("Tavoitteena on tehdä maali potkaisemalla pallo verkkoon"),
        ]
        result = await evaluate_db_data(
            transcript=transcript,
            vector_database_response=vectors,
            thinking_context="User mentioned football player count.",
            query_history=[],
        )
        assert result["status"] == "found", (
            f"Flash should return new info (90 min, scoring). Got: {result}"
        )
        assert "11" not in result["information"] or "90" in result["information"], (
            f"Flash must not just repeat the 11-player fact. Got: {result['information']}"
        )


# ---------------------------------------------------------------------------
# CASE 8: Tool spam — same answer must not be returned twice via query_history
# ---------------------------------------------------------------------------

class TestToolSpam:

    @pytest.mark.asyncio
    async def test_same_answer_not_returned_again_different_query(self):
        """
        Flash already returned "Pepperoni on klassinen pizzatäyte" via a previous query.
        A new tool call comes in with a slightly different query but same vectors.
        Flash must return not_relevant.
        """
        transcript = "Tilasin pizzan pepperonilla. Mitä muita täytteitä on?"
        vectors = [
            make_vector("Pepperoni on klassinen pizzatäyte"),
            make_vector("Juusto on pizzan tärkein ainesosa"),
        ]
        query_history = [
            {
                "query": "pepperoni pizza",
                "thinking_context": "User mentioned pepperoni pizza.",
                "answer": "Pepperoni on klassinen pizzatäyte.",
            }
        ]
        result = await evaluate_db_data(
            transcript=transcript,
            vector_database_response=vectors,
            thinking_context="User asked about pizza toppings again.",
            query_history=query_history,
        )
        assert result["status"] == "not_relevant", (
            f"Flash should not repeat pepperoni info already sent. Got: {result}"
        )


# ---------------------------------------------------------------------------
# CASE 9: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_empty_thinking_context(self):
        """
        Empty thought_context — Flash should return not_relevant
        as the context is not grounded in anything.
        """
        transcript = "Hei miten menee."
        vectors = [
            make_vector("Toyota Yaris tunnetaan polttoainetehokkuudestaan"),
        ]
        result = await evaluate_db_data(
            transcript=transcript,
            vector_database_response=vectors,
            thinking_context="",
            query_history=[],
        )
        assert result["status"] == "not_relevant", (
            f"Empty thinking_context should return not_relevant. Got: {result}"
        )

    @pytest.mark.asyncio
    async def test_vague_user_question_unrelated_vectors(self):
        """
        User asks vaguely "kerro lisää".
        Vectors contain unrelated car information.
        Flash should return not_relevant.
        """
        transcript = "Jalkapallosta puhuttiin. Kerro lisää."
        vectors = [
            make_vector("Toyota Yaris on polttoainetehokas"),
            make_vector("Punaiset urheiluautot herättävät huomiota"),
        ]
        result = await evaluate_db_data(
            transcript=transcript,
            vector_database_response=vectors,
            thinking_context="User asked for more information generally.",
            query_history=[],
        )
        assert result["status"] == "not_relevant", (
            f"Unrelated vectors should not be returned for vague football query. Got: {result}"
        )

    @pytest.mark.asyncio
    async def test_vector_matches_wrong_part_of_context(self):
        """
        Transcript mentions both football and pizza.
        Vector is about pizza but thinking_context is about football.
        Flash should return not_relevant — vector doesn't answer the thought_context.
        """
        transcript = "Jalkapallosta puhuttiin. Söin myös pizzaa tänään."
        vectors = [
            make_vector("Pepperoni on klassinen pizzatäyte"),
            make_vector("Juusto on pizzan tärkein ainesosa"),
        ]
        result = await evaluate_db_data(
            transcript=transcript,
            vector_database_response=vectors,
            thinking_context="User mentioned football and might have context about football rules.",
            query_history=[],
        )
        assert result["status"] == "not_relevant", (
            f"Pizza vectors should not answer football thought_context. Got: {result}"
        )
