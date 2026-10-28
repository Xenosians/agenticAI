"""KiCad netlist inspection (FR-093 PCB/EDA): bounded parsing and deterministic design checks."""
import hashlib

import pytest

from services.engineering.netlist import NetlistError, inspect_netlist

KICAD_7 = """(export (version "E")
  (design
    (source "/home/designer/private/board.kicad_sch")
    (date "2026-10-29T01:00:00")
    (tool "Eeschema 7.0.10"))
  (components
    (comp (ref "R1")
      (value "10k")
      (footprint "Resistor_SMD:R_0603_1608Metric")
      (libsource (lib "Device") (part "R") (description "Resistor")))
    (comp (ref "C1")
      (value "100nF")
      (footprint "Capacitor_SMD:C_0603_1608Metric")
      (libsource (lib "Device") (part "C") (description "Unpolarized capacitor")))
    (comp (ref "U1")
      (value "MCU \\"rev B\\"")
      (footprint "Package_QFP:LQFP-48_7x7mm_P0.5mm")
      (libsource (lib "MCU") (part "STM32") (description "Microcontroller"))))
  (nets
    (net (code "1") (name "GND")
      (node (ref "C1") (pin "2") (pintype "passive"))
      (node (ref "U1") (pin "8") (pinfunction "VSS") (pintype "power_in")))
    (net (code "2") (name "+3V3")
      (node (ref "C1") (pin "1") (pintype "passive"))
      (node (ref "R1") (pin "1") (pintype "passive"))
      (node (ref "U1") (pin "1") (pinfunction "VDD") (pintype "power_in")))
    (net (code "3") (name "/RESET")
      (node (ref "R1") (pin "2") (pintype "passive"))
      (node (ref "U1") (pin "7") (pinfunction "NRST") (pintype "input")))))
"""

KICAD_5 = """(export (version D)
  (design (source /tmp/x.sch) (tool "Eeschema 5.1.12"))
  (components
    (comp (ref J1) (value Conn_01x02) (footprint Connector:PinHeader_1x02)
      (libsource (lib Connector) (part Conn_01x02)))
    (comp (ref D1) (value LED) (footprint LED_SMD:LED_0603)))
  (nets
    (net (code 1) (name GND) (node (ref J1) (pin 2)) (node (ref D1) (pin 1)))
    (net (code 2) (name "Net-(D1-Pad2)") (node (ref D1) (pin 2)) (node (ref J1) (pin 1)))))
"""


def codes(report):
    return sorted((finding["code"], finding["subject"]) for finding in report["findings"])


def test_a_clean_kicad_7_netlist_reports_parts_nets_and_no_findings():
    report = inspect_netlist(KICAD_7)
    assert report["schema"] == "pcb-netlist-inspection.v1"
    assert report["state_kind"] == "desired" and report["actuation"] == "none"
    assert report["format"] == "kicad" and report["tool"] == "Eeschema 7.0.10"
    assert report["sha256"] == hashlib.sha256(KICAD_7.encode()).hexdigest()
    assert report["summary"] == {"components": 3, "nets": 3, "connections": 7, "errors": 0, "warnings": 0}
    assert report["components"][0] == {"ref": "C1", "value": "100nF", "footprint": "Capacitor_SMD:C_0603_1608Metric",
                                      "part": "Device:C", "pins": 2}
    u1 = [row for row in report["components"] if row["ref"] == "U1"][0]
    assert u1["value"] == 'MCU "rev B"'
    assert report["nets"][0] == {"code": "1", "name": "GND", "connections": 2, "nodes": ["C1.2", "U1.8"]}
    assert report["findings"] == []


def test_the_designer_file_path_is_never_echoed():
    assert "/home/designer" not in str(inspect_netlist(KICAD_7))


def test_kicad_5_bare_atoms_are_supported():
    report = inspect_netlist(KICAD_5)
    assert report["tool"] == "Eeschema 5.1.12"
    assert [row["ref"] for row in report["components"]] == ["D1", "J1"]
    assert report["components"][1]["part"] == "Connector:Conn_01x02"
    assert report["components"][0]["part"] is None
    assert report["nets"][1]["name"] == "Net-(D1-Pad2)"


def test_design_findings_are_deterministic():
    text = KICAD_7.replace('(footprint "Resistor_SMD:R_0603_1608Metric")', "").replace(
        '(node (ref "R1") (pin "2") (pintype "passive"))', "")
    text = text.replace("(nets", """(nets
    (net (code "9") (name "unconnected-(U1-Pad3)") (node (ref "U1") (pin "3") (pintype "input")))
    (net (code "10") (name "/SPARE") (node (ref "U1") (pin "4") (pintype "input")))
    (net (code "11") (name "/LOOP") (node (ref "U1") (pin "8") (pintype "input")) (node (ref "Q9") (pin "1")))
    (net (code "12") (name "/PULL") (node (ref "R?") (pin "1")) (node (ref "U1") (pin "5")))""", 1)
    text = text.replace("(components", """(components
    (comp (ref "H1") (value "MountingHole") (footprint "MountingHole:MountingHole_3.2mm_M3"))
    (comp (ref "R1") (value "1k") (footprint "Resistor_SMD:R_0402_1005Metric"))
    (comp (ref "R?") (value "1k") (footprint "Resistor_SMD:R_0402_1005Metric"))""", 1)
    report = inspect_netlist(text)
    assert codes(report) == [
        ("duplicate_reference", "R1"),
        ("missing_footprint", "R1"),
        ("no_connections", "H1"),
        ("pin_on_multiple_nets", "U1.8"),
        ("single_connection_net", "/RESET"),
        ("single_connection_net", "/SPARE"),
        ("unannotated_reference", "R?"),
        ("unconnected_pin", "unconnected-(U1-Pad3)"),
        ("unknown_component", "Q9"),
    ]
    severities = {finding["code"]: finding["severity"] for finding in report["findings"]}
    assert severities["missing_footprint"] == "error" and severities["duplicate_reference"] == "error"
    assert severities["no_connections"] == "info" and severities["unconnected_pin"] == "info"
    assert severities["single_connection_net"] == "warning"
    assert report["summary"]["errors"] == 5 and report["summary"]["warnings"] == 2


@pytest.mark.parametrize("text", [
    "", "   ", "(export", "(export))", "export (version E)", "(design (tool x))",
    "(export (version \"E\") (components (comp (ref)))) ", '(export (version "E")) (extra)',
    "(export (version \"unterminated))",
])
def test_malformed_netlists_are_refused(text):
    with pytest.raises(NetlistError) as caught:
        inspect_netlist(text)
    assert caught.value.code == "invalid_netlist"


def test_input_size_and_nesting_are_bounded():
    with pytest.raises(NetlistError):
        inspect_netlist("(export " + "x " * 1_100_000 + ")")
    with pytest.raises(NetlistError):
        inspect_netlist("(export " + "(a " * 100 + ")" * 101)


def test_component_and_net_counts_are_bounded():
    comps = "".join(f'(comp (ref "R{i}") (value "1k") (footprint "R"))' for i in range(5001))
    with pytest.raises(NetlistError):
        inspect_netlist(f'(export (version "E") (components {comps}) (nets))')


def test_large_nets_list_a_bounded_number_of_nodes():
    comps = "".join(f'(comp (ref "R{i}") (value "1k") (footprint "R"))' for i in range(80))
    nodes = "".join(f'(node (ref "R{i}") (pin "1"))' for i in range(80))
    report = inspect_netlist(f'(export (version "E") (components {comps}) (nets (net (code "1") (name "GND") {nodes})))')
    assert report["nets"][0]["connections"] == 80 and len(report["nets"][0]["nodes"]) == 50


def test_text_fields_are_single_line_and_bounded():
    text = KICAD_7.replace('(value "10k")', '(value "' + "v" * 400 + '\\nnext")')
    value = [row for row in inspect_netlist(text)["components"] if row["ref"] == "R1"][0]["value"]
    assert "\n" not in value and len(value) == 120
