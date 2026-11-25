"""Name parsing on real ISLRTC / INCLUDE names (docs/ISL_DESIGN.md 6.1). WordNet is switched off for the
deterministic cases (CI has no WordNet data); synonym-list detection with WordNet is tested separately."""

from __future__ import annotations

import pytest

from app.isl.vocab import wordnet as wnx
from app.isl.vocab.normalize import (
    apply_overrides,
    clean_text,
    load_overrides,
    match_key,
    parse_include_word,
    parse_islrtc_name,
    parse_region,
    slugify,
)


@pytest.fixture
def no_wordnet():
    wnx.disable(True)
    yield
    wnx.disable(False)


# (path, concept_id, kind, sign_no, region, domain, number_value, required flags)
ISLRTC_CASES = [
    ("P/Pencil.mp4", "isl.pencil", "word", 1, None, None, None, []),
    ("P/Pencil_(Sign_2).mp4", "isl.pencil", "word", 2, None, None, None, []),
    ("P/Pencil_(Sign_3).mp4", "isl.pencil", "word", 3, None, None, None, []),
    ("C/Can_(Sign_2).mp4", "isl.can", "word", 2, None, None, None, []),
    ("B/Bharat (Sign 4).mp4", "isl.bharat", "word", 4, None, None, None, []),
    ("L/Language_Acquisition_(Sign).mp4", "isl.language-acquisition", "multiword", 1, None, None, None, []),
    ("P/Public_Limited_Company (1).mp4", "isl.public-limited-company", "multiword", 1, None, None, None, ["bare_number_marker"]),
    ("A/Abdicate_(Sign_2).mp4", "isl.abdicate", "word", 2, None, None, None, []),
    ("K/Knife_(Sign_2).mp4", "isl.knife", "word", 2, None, None, None, []),
    ("C/Cyber_Crime_(Sign_2).mp4", "isl.cyber-crime", "multiword", 2, None, None, None, []),
    ("T/Transcendentalism - English.mp4", "isl.transcendentalism", "word", 1, None, None, None, ["english_marker"]),
    ("A/AB Testing- English.mp4", "isl.ab-testing", "multiword", 1, None, None, None, ["english_marker"]),
    ("A/Adaptable - English.mp4", "isl.adaptable", "word", 1, None, None, None, ["english_marker"]),
    ("A/Air Quality Index (AQI) - English.mp4", "isl.air-quality-index", "multiword", 1, None, None, None,
     ["english_marker", "acronym"]),
    ("A/Able_(Explanation).mp4", "isl.able", "explanation", 1, None, None, None, []),
    ("B/Baggage_Reclaim_(Explanation)_(Sign_2).mp4", "isl.baggage-reclaim", "explanation", 2, None, None, None, []),
    ("A/Abundant_(Sign_and_Explanation).mp4", "isl.abundant", "explanation", 1, None, None, None, ["sign_and_explanation"]),
    ("C/Cartoon_(Explantion).mp4", "isl.cartoon", "explanation", 1, None, None, None, ["typo_fixed"]),
    ("B/Buffet_(Example).mp4", "isl.buffet", "example", 1, None, None, None, []),
    ("A/Area_(Math)_(Explanation).mp4", "isl.area~math", "explanation", 1, None, "math", None, []),
    ("O/Over_(Indore_Madhya_Pradesh).mp4", "isl.over", "word", 1, "Indore, Madhya Pradesh", None, None, []),
    ("T/Test_(Indore).mp4", "isl.test", "word", 1, "Indore, Madhya Pradesh", None, None, []),
    ("P/Pace_Bowling_Fast_Bowling_(Thiruvananthapuram_Kerala).mp4", "isl.pace-bowling-fast-bowling", "multiword", 1,
     "Thiruvananthapuram, Kerala", None, None, []),
    ("M/Madhya_Pradesh.mp4", "isl.madhya-pradesh", "multiword", 1, None, None, None, []),
    ("B/Breathing_Rate_(Academic).mp4", "isl.breathing-rate~academic", "multiword", 1, None, "academic", None, []),
    ("I/Iron_Pillar_(Academic).mp4", "isl.iron-pillar~academic", "multiword", 1, None, "academic", None, []),
    ("A/AABBA_(Rhyme_Scheme).mp4", "isl.aabba~rhyme-scheme", "word", 1, None, "rhyme scheme", None, []),
    ("L/Light_(Electricity).mp4", "isl.light~electricity", "word", 1, None, "electricity", None, []),
    ("S/Second_(Time).mp4", "isl.second~time", "word", 1, None, "time", None, []),
    ("T/Tight_(Clothes)_(Sign_2).mp4", "isl.tight~clothes", "word", 2, None, "clothes", None, []),
    ("Numbers/12_Twelve.mp4", "isl.twelve", "number", 1, None, None, 12, []),
    ("Numbers/1_One.mp4", "isl.one", "number", 1, None, None, 1, []),
    ("Numbers/100000_One_Lakh.mp4", "isl.one-lakh", "number", 1, None, None, 100000, []),
    ("Numbers/10000000_One_Crore.mp4", "isl.one-crore", "number", 1, None, None, 10000000, []),
    ("Numbers/21_Twenty_One.mp4", "isl.twenty-one", "number", 1, None, None, 21, []),
    ("Numbers/120 Volt.mp4", "isl.120-volt", "multiword", 1, None, None, None, []),
    ("Numbers/3G.mp4", "isl.3g", "word", 1, None, None, None, []),
    ("Numbers/404 error.mp4", "isl.404-error", "multiword", 1, None, None, None, []),
    ("Numbers/Numbers_1-10_(Odisha).mp4", "isl.numbers-1-10", "compilation", 1, "Odisha", None, None, []),
    ("A/ATM.mp4", "isl.atm", "word", 1, None, None, None, []),
    ("A/AC_to_DC_Converter.mp4", "isl.ac-to-dc-converter", "multiword", 1, None, None, None, []),
    ("A/A-roll.mp4", "isl.a-roll", "word", 1, None, None, None, []),
    ("H/High_Five.mp4", "isl.high-five", "multiword", 1, None, None, None, []),
    ("S/Shut_Up_Stop_Talking_(Sign_2).mp4", "isl.shut-up-stop-talking", "multiword", 2, None, None, None, []),
    ("C/Current_and_Ampere.mp4", "isl.current-and-ampere", "multiword", 1, None, None, None, []),
    ("A/A_bird_in_the_hand_is_worth_two_in_the_bush.mp4", "isl.a-bird-in-the-hand-is-worth-two-in-the-bush", "phrase",
     1, None, None, None, []),
    ("C/Crop_Rotation_in_Vegetable_Production.mp4", "isl.crop-rotation-in-vegetable-production", "phrase", 1, None,
     None, None, []),
    ("A/Abhor Despise Detest Hate Loathe.mp4", "isl.abhor-despise-detest-hate-loathe", "phrase", 1, None, None, None, []),
    ("A/Ab (Symbol of Line) (Math).mp4", "isl.ab~symbol-of-line", "word", 1, None, "symbol of line", None, []),
    ("D/Diwali.mp4", "isl.diwali", "word", 1, None, None, None, []),
    ("V/Vadodara.mp4", "isl.vadodara", "word", 1, None, None, None, []),
    ("A/Agro_based_industry.mp4", "isl.agro-based-industry", "multiword", 1, None, None, None, []),
    ("I/Inter-Departmental_Note.mp4", "isl.inter-departmental-note", "multiword", 1, None, None, None, []),
    ("H/High_Dependency_Unit_(HDU).mp4", "isl.high-dependency-unit", "multiword", 1, None, None, None, ["acronym"]),
    ("S/Second_in_Command_(SIC).mp4", "isl.second-in-command", "multiword", 1, None, None, None, ["acronym"]),
    ("R/Race_(Social_Science).mp4", "isl.race~social-science", "word", 1, None, "social science", None, []),
    ("A/Advance_(Administrator).mp4", "isl.advance~administrator", "word", 1, None, "administrator", None, []),
    ("A/Area_(Maths).mp4", "isl.area~math", "word", 1, None, "math", None, []),
]

INCLUDE_CASES = [
    ("63. Winter", "isl.winter", "word", None, []),
    ("1. loud", "isl.loud", "word", None, []),
    ("Store or Shop", "isl.store", "synonym_list", None, ["store", "shop"]),
    ("20. Street or Road", "isl.street", "synonym_list", None, ["street", "road"]),
    ("Second (Number)", "isl.second~number", "word", "number", []),
    ("46. you (plural)", "isl.you~plural", "word", "plural", []),
    ("8. Race (ethnicity)", "isl.race~ethnicity", "word", "ethnicity", []),
    ("Ex. Monsoon", "isl.monsoon", "word", None, []),
    ("42. T-Shirt", "isl.t-shirt", "word", None, []),
    ("54. Cell phone", "isl.cell-phone", "multiword", None, []),
    ("49. How are you", "isl.how-are-you", "multiword", None, []),
    ("21. Train Station", "isl.train-station", "multiword", None, []),
]


@pytest.mark.parametrize("path,cid,kind,sign_no,region,domain,number,flags", ISLRTC_CASES)
def test_parse_islrtc_name(no_wordnet, path, cid, kind, sign_no, region, domain, number, flags):
    pn = parse_islrtc_name(path)
    assert pn.concept_id == cid
    assert pn.kind == kind
    assert pn.sign_no == sign_no
    assert pn.region == region
    assert pn.domain == domain
    assert pn.number_value == number
    for f in flags:
        assert f in pn.flags


@pytest.mark.parametrize("word_dir,cid,kind,sense,syn", INCLUDE_CASES)
def test_parse_include_word(no_wordnet, word_dir, cid, kind, sense, syn):
    pn = parse_include_word(word_dir)
    assert pn.concept_id == cid
    assert pn.kind == kind
    assert pn.sense == sense
    if syn:
        assert [s.lower() for s in pn.synonyms] == syn


def test_case_count():
    assert len(ISLRTC_CASES) + len(INCLUDE_CASES) >= 60


def test_basics():
    assert clean_text("  Abhor_Despise   Detest ") == "Abhor Despise Detest"
    assert slugify("AC to DC Converter") == "ac-to-dc-converter"
    assert slugify("Café (Sign 2)") == "cafe-sign-2"
    assert match_key("T-Shirt") == match_key("t shirt") == "tshirt"
    assert parse_region("Indore") == "Indore, Madhya Pradesh"
    assert parse_region("Indore Madhya Pradesh") == "Indore, Madhya Pradesh"
    assert parse_region("Andhra_Pradesh_Telangana") == "Andhra Pradesh, Telangana"
    assert parse_region("Orissa") == "Odisha"
    assert parse_region("Academic") is None


def test_acronyms_stay_upper(no_wordnet):
    assert parse_islrtc_name("A/ATM.mp4").headword == "ATM"
    assert parse_islrtc_name("A/AB Testing- English.mp4").headword == "AB testing"
    pn = parse_islrtc_name("A/Air Quality Index (AQI) - English.mp4")
    assert "AQI" in pn.synonyms


def test_no_wordnet_flag(no_wordnet):
    assert "no_wordnet" in parse_islrtc_name("M/Mineral_Matter.mp4").flags


def test_overrides_win(no_wordnet):
    ov = load_overrides()
    pn = apply_overrides(parse_islrtc_name("A/Alphabet.mp4"), "islrtc/A/Alphabet.mp4", ov)
    assert pn.kind == "compilation" and "override" in pn.flags
    pn = apply_overrides(parse_islrtc_name("Numbers/Numbers_(0-1_Crore).mp4"), "islrtc/Numbers/Numbers_(0-1_Crore).mp4", ov)
    assert pn.kind == "compilation"
    custom = [{"match": r"^islrtc/X/Foo\.mp4$", "set": {"headword": "bar", "flags_add": ["typo_fixed"]}}]
    pn = apply_overrides(parse_islrtc_name("X/Foo.mp4"), "islrtc/X/Foo.mp4", custom)
    assert pn.headword == "bar" and "typo_fixed" in pn.flags


@pytest.mark.skipif(wnx.get_wordnet() is None, reason="WordNet data not installed (python -m app.isl.cli wordnet)")
def test_synonym_lists_with_wordnet():
    for path, syn in [
        ("A/Abhor Despise Detest Hate Loathe.mp4", ["abhor", "despise", "detest", "hate", "loathe"]),
        ("A/Afraid_Scared.mp4", ["afraid", "scared"]),
        ("A/Alike_similar.mp4", ["alike", "similar"]),
        ("D/Defamation_Libel_Slander.mp4", ["defamation", "libel", "slander"]),
    ]:
        pn = parse_islrtc_name(path)
        assert pn.kind == "synonym_list", path
        assert [s.lower() for s in pn.synonyms] == syn
        assert pn.headword.lower() == syn[0]
    for path in ("M/Mineral_Matter.mp4", "H/Hero_Honda.mp4", "S/Set_Square.mp4", "R/Run_Out.mp4", "H/High_Five.mp4",
                 "S/Small_Intestine_Cancer.mp4", "P/Positive_Terminal.mp4"):
        assert parse_islrtc_name(path).kind == "multiword", path
    big = parse_include_word("83. big large")
    assert big.kind == "synonym_list" and big.concept_id == "isl.big"
