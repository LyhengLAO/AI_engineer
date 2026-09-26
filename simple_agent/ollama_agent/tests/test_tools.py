from agent.tools import REGISTRY


def test_calculator_addition():
    assert REGISTRY["calculatrice"].func("2+2") == "4"


def test_calculator_invalid_expression():
    result = REGISTRY["calculatrice"].func("import os")
    assert result.startswith("Erreur")


def test_word_count():
    assert REGISTRY["compte_mots"].func("un deux trois") == "3"


def test_reverse_text():
    assert REGISTRY["inverse_texte"].func("abc") == "cba"
