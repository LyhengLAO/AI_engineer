from agent.core import parse_response


def test_parse_action():
    kind, name, arg = parse_response("ACTION: calculatrice(2+2)")
    assert kind == "action"
    assert name == "calculatrice"
    assert arg == "2+2"


def test_parse_final():
    kind, payload, extra = parse_response("FINAL: la réponse est 4")
    assert kind == "final"
    assert payload == "la réponse est 4"
    assert extra is None


def test_parse_fallback_treats_unformatted_text_as_final():
    kind, payload, _ = parse_response("Je ne sais pas trop quoi répondre")
    assert kind == "final"
