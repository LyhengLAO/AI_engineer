from agent.tools import TOOLS

TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def test_calculator_addition():
    assert TOOLS_BY_NAME["calculatrice"].func("2+2") == "4"


def test_calculator_invalid_expression():
    result = TOOLS_BY_NAME["calculatrice"].func("import os")
    assert result.startswith("Erreur")


def test_word_count():
    assert TOOLS_BY_NAME["compte_mots"].func("un deux trois") == "3"


def test_reverse_text():
    assert TOOLS_BY_NAME["inverse_texte"].func("abc") == "cba"


def test_all_tools_registered():
    expected = {"calculatrice", "compte_mots", "inverse_texte", "recherche_web"}
    assert expected.issubset(TOOLS_BY_NAME.keys())
