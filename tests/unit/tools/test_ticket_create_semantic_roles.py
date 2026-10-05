from tools.ticketing.catalog import TICKETING_TOOLS


def test_ticket_create_declares_required_grounded_and_derived_roles():
    tool = TICKETING_TOOLS["ticket_create"]
    assert tool["required_arguments"] == ["project_key", "summary"]
    assert tool["grounded_arguments"] == ["project_key", "ticket_type"]
    assert tool["derived_arguments"] == ["summary"]
    assert set(tool["grounded_arguments"]) & set(tool["derived_arguments"]) == set()
