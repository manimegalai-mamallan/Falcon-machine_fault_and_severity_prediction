"""
Plain-English explanations and maintenance recommendations.

Static lookup table -- driven by the predicted (fault, severity) regardless
of which model produced the prediction.
"""
from __future__ import annotations


PLAIN_ENGLISH = {
    ("Normal", None): (
        "The machine is operating within normal vibration limits. "
        "No fault signatures were detected; routine monitoring is sufficient."
    ),
    ("Unbalance", "Low"): (
        "A minor mass imbalance was detected on the rotating shaft. "
        "This typically causes slight extra wear on bearings but is not urgent."
    ),
    ("Unbalance", "Medium"): (
        "A noticeable mass imbalance is present on the rotating shaft. "
        "Vibration is elevated; bearing life may be reduced if left uncorrected."
    ),
    ("Unbalance", "High"): (
        "Significant mass imbalance detected. Machine is operating with "
        "high vibration; bearing damage and energy loss are likely if not "
        "corrected promptly."
    ),
    ("Misalignment", "Low"): (
        "A minor shaft alignment deviation was detected between coupled "
        "components. Not urgent, but should be checked at the next "
        "scheduled service."
    ),
    ("Misalignment", "Medium"): (
        "Noticeable shaft misalignment between the motor and driven equipment. "
        "Coupling and bearing wear is accelerating; correction is recommended."
    ),
    ("Misalignment", "High"): (
        "Significant shaft misalignment detected. Coupling damage, premature "
        "bearing failure, and seal leaks become increasingly likely without "
        "correction."
    ),
    ("Looseness", "Low"): (
        "Minor mechanical looseness detected. Likely a slightly under-torqued "
        "mounting bolt or a small clearance issue; routine tightening "
        "recommended."
    ),
    ("Looseness", "Medium"): (
        "Mechanical looseness is causing measurable extra vibration. "
        "Inspect mounting bolts, foundation, and bearing housings; "
        "correction before further degradation is recommended."
    ),
    ("Looseness", "High"): (
        "Severe mechanical looseness detected. Vibration is high and may be "
        "damaging the foundation or housing; machine should be inspected and "
        "re-secured promptly."
    ),
}


ACTIONS = {
    ("Normal", None): {
        "priority": "none",
        "recommended_check":
            "No action required. Continue routine monitoring on the regular "
            "maintenance schedule.",
    },

    ("Unbalance", "Low"): {
        "priority": "scheduled",
        "recommended_check":
            "Inspect rotor for fouling, dirt buildup, or minor mass deposits "
            "at the next scheduled service. Consider rebalancing if vibration "
            "trends upward.",
    },
    ("Unbalance", "Medium"): {
        "priority": "soon",
        "recommended_check":
            "Schedule rotor balancing within 2-4 weeks. Inspect rotor for "
            "visible damage, cracked vanes, or deposits before balancing. "
            "Verify bearing condition.",
    },
    ("Unbalance", "High"): {
        "priority": "immediate",
        "recommended_check":
            "Schedule rotor balancing as soon as practical (within 1 week). "
            "Inspect for cracked vanes, broken blades, or shifted balance "
            "weights. Check bearing temperatures before continued operation.",
    },

    ("Misalignment", "Low"): {
        "priority": "scheduled",
        "recommended_check":
            "Verify shaft alignment with a dial indicator or laser alignment "
            "tool at the next service. Check coupling for visible wear.",
    },
    ("Misalignment", "Medium"): {
        "priority": "soon",
        "recommended_check":
            "Realign motor-to-driven-equipment coupling within 2-3 weeks. "
            "Inspect coupling element for wear; replace if cracked or "
            "deformed. Verify base bolt torque.",
    },
    ("Misalignment", "High"): {
        "priority": "immediate",
        "recommended_check":
            "Realign coupling as soon as practical (within 1 week). Inspect "
            "coupling for damage. Check bearing seals for leakage. Verify "
            "thermal growth allowances if hot alignment is needed.",
    },

    ("Looseness", "Low"): {
        "priority": "scheduled",
        "recommended_check":
            "Verify foundation bolt torque and bearing housing fasteners at "
            "the next scheduled service. Inspect grout or shim condition.",
    },
    ("Looseness", "Medium"): {
        "priority": "soon",
        "recommended_check":
            "Within 2-3 weeks: re-torque all base bolts and bearing housing "
            "fasteners to spec. Inspect for cracked grout, worn shims, or "
            "elongated bolt holes. Check bearing internal clearance.",
    },
    ("Looseness", "High"): {
        "priority": "immediate",
        "recommended_check":
            "Inspect within 1 week. Re-torque all fasteners. Check for "
            "cracked grout, foundation damage, or excessive bearing internal "
            "clearance. Replace damaged components before continued operation.",
    },
}


def get_plain_english(fault: str, severity: str | None) -> str:
    key = (fault, severity)
    if key in PLAIN_ENGLISH:
        return PLAIN_ENGLISH[key]
    return f"{fault} detected" + (
        f" at {severity} severity." if severity else "."
    )


def get_action(fault: str, severity: str | None) -> dict:
    key = (fault, severity)
    if key in ACTIONS:
        return dict(ACTIONS[key])
    return {
        "priority":         "soon",
        "recommended_check": f"Inspect machine for {fault} symptoms.",
    }
