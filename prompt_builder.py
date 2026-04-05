"""
Prompt building and text normalization for SmartPharma RAG.
Handles context assembly and response post-processing.
"""
import re
import os
from typing import List, Dict, Any
from config import SYSTEM_PRIMER


# =========================
# Drug classification for relevance checking
# =========================
ANTICOAGULANT_DRUGS = {
    "enoxaparin", "heparin", "warfarin", "dabigatran", "rivaroxaban",
    "apixaban", "edoxaban", "fondaparinux", "dalteparin", "tinzaparin",
    "betrixaban", "nadroparin", "bemiparin"
}

ANTIMICROBIAL_DIAGNOSES = {
    "pneumonia", "community-acquired pneumonia", "cap", "hospital-acquired pneumonia",
    "hap", "sepsis", "septicaemia", "urinary tract infection", "uti",
    "skin and soft tissue infection", "ssti", "cellulitis", "meningitis",
    "endocarditis", "osteomyelitis", "intra-abdominal infection",
    "tuberculosis", "tb", "bronchitis", "pharyngitis", "tonsillitis",
    "otitis media", "sinusitis", "peritonitis", "cholangitis",
    "pyelonephritis", "cystitis", "abscess", "wound infection",
    "surgical site infection", "bacteremia", "fungal infection",
    "candidiasis", "aspergillosis", "malaria", "dengue"
}

ANTICOAGULANT_INDICATIONS = {
    "dvt", "deep vein thrombosis", "pe", "pulmonary embolism",
    "vte", "venous thromboembolism", "atrial fibrillation", "af",
    "stroke prevention", "mechanical heart valve", "prosthetic valve",
    "thromboprophylaxis", "thrombosis", "embolism",
    "left ventricular thrombus", "lvt", "acs", "acute coronary syndrome",
    "nstemi", "stemi", "myocardial infarction"
}


def check_drug_relevance(
    medicines: list,
    retrieved: list,
    diagnosis: str,
    per_drug_results: Dict[str, Dict[str, Any]] = None
) -> str:
    """
    For each prescribed drug, check whether it is indicated for the given diagnosis.
    Uses:
      1. Drug classification (anticoagulant vs antimicrobial)
      2. Whether the drug appears in retrieved chunks relevant to the diagnosis
      3. Per-drug retrieval results if available
    """
    if not medicines:
        return ""

    hints = []
    diagnosis_lower = diagnosis.lower().strip()

    # Determine if diagnosis is primarily infectious
    is_infection_diagnosis = any(
        dx in diagnosis_lower for dx in ANTIMICROBIAL_DIAGNOSES
    )

    # Determine if diagnosis has anticoagulant indication
    has_anticoag_indication = any(
        ind in diagnosis_lower for ind in ANTICOAGULANT_INDICATIONS
    )

    for med in medicines:
        name = med.get("name", "").strip()
        if not name:
            continue

        name_lower = name.lower()

        # Check if drug is an anticoagulant
        is_anticoagulant = name_lower in ANTICOAGULANT_DRUGS

        # Check if drug appears in the diagnosis-retrieved chunks
        found_in_diagnosis_chunks = any(
            name_lower in item["text"].lower()
            for item in retrieved
        )

        # Check per-drug retrieval results if available
        per_drug_found = False
        per_drug_source = None
        if per_drug_results and name in per_drug_results:
            per_drug_found = per_drug_results[name]["found"]
            per_drug_source = per_drug_results[name].get("source_type")

        # Decision logic — keep hints SHORT for 8B model
        if is_anticoagulant and is_infection_diagnosis and not has_anticoag_indication:
            hints.append(
                f"{name}: NOT INDICATED. This is an anticoagulant — not for '{diagnosis}'. "
                f"Flag as prescription error. Do NOT assess dose."
            )
        elif is_anticoagulant and has_anticoag_indication:
            hints.append(f"{name}: INDICATED. Anticoagulant matches indication. Verify dose.")
        elif not is_anticoagulant and found_in_diagnosis_chunks:
            hints.append(f"{name}: INDICATED. Found in guidelines for '{diagnosis}'. Verify dose.")
        elif not is_anticoagulant and per_drug_found:
            hints.append(f"{name}: POSSIBLY INDICATED. Found in guidelines but not for '{diagnosis}'.")
        elif not is_anticoagulant and not found_in_diagnosis_chunks and not per_drug_found:
            hints.append(f"{name}: NOT FOUND in guidelines. Cannot determine indication.")
        else:
            hints.append(f"{name}: UNCERTAIN relevance to '{diagnosis}'.")

    return "\n".join(hints)


def check_dose_in_range(medicines: list, retrieved: list) -> str:
    hints = []

    for med in medicines:
        name = med.get("name", "")
        dose_str = med.get("dose", "")
        unit = med.get("unit", "mg")

        try:
            dose_val = float(dose_str)
            if unit.lower() == 'g':
                dose_val = dose_val * 1000
        except (ValueError, TypeError):
            continue

        range_found = None
        for item in retrieved:
            text = item.get("text", "")
            pattern = re.search(
                rf'{re.escape(name)}[^.]*?(\d+)(?:\s*[-–]\s*(\d+))?\s*mg',
                text, re.IGNORECASE
            )
            if pattern:
                min_dose = float(pattern.group(1))
                max_dose = float(pattern.group(2)) if pattern.group(2) else min_dose
                range_found = (min_dose, max_dose)
                break

        if range_found:
            min_d, max_d = range_found
            within = min_d <= dose_val <= max_d
            if within:
                hints.append(
                    f"DOSE VERDICT: {name} {dose_str}{unit} = {dose_val:.0f}mg. "
                    f"Guideline range found: {min_d:.0f}{f'-{max_d:.0f}' if min_d != max_d else ''}mg. "
                    f"Dose AMOUNT is within range. You MUST now evaluate if the FREQUENCY is correct according to the guidelines."
                )
            else:
                hints.append(
                    f"DOSE VERDICT: {name} {dose_str}{unit} = {dose_val:.0f}mg. "
                    f"Guideline range found: {min_d:.0f}{f'-{max_d:.0f}' if min_d != max_d else ''}mg. "
                    f"Dose AMOUNT is OUTSIDE RANGE. You MUST state this and evaluate if the FREQUENCY is correct according to the guidelines."
                )
        else:
            hints.append(
                f"DOSE PRE-CHECK: {name} = {dose_val:.0f}mg. "
                f"No specific range found in retrieved evidence — state 'Cannot determine'."
            )

    return "\n".join(hints)


def normalize_dose_to_mg(dose: str, unit: str) -> str:
    try:
        val = float(dose)
        if unit.lower() == 'g':
            return f"{dose}{unit} ({int(val * 1000)}mg equivalent)"
        return f"{dose}{unit}"
    except (ValueError, TypeError):
        return f"{dose}{unit}"


def make_prompt(
    question: str,
    retrieved: List[Dict[str, Any]],
    patient: Dict[str, Any] = None,
    labs: Dict[str, Any] = None,
    vitals: Dict[str, Any] = None,
    diagnosis: str = "",
    medicines: List[Dict[str, Any]] = None,
    per_drug_results: Dict[str, Dict[str, Any]] = None
) -> str:
    patient = patient or {}
    labs = labs or {}
    vitals = vitals or {}
    medicines = medicines or []

    patient_section = ""
    if patient:
        patient_section = "\nPATIENT PROFILE:\n"
        for key, value in patient.items():
            if value:
                formatted_key = key.replace("_", " ").title()
                patient_section += f"- {formatted_key}: {value}\n"

    age_val = patient.get("age") or patient.get("Age")
    try:
        if age_val and int(age_val) < 18:
            patient_section += (
                "\n⚠️ NOTE: This patient is a minor. "
                "Only adult guidelines are currently loaded. "
                "Verify with a paediatrician before acting on these recommendations.\n"
            )
    except (ValueError, TypeError):
        pass

    allergy_val = (
        patient.get("allergy_details") or
        patient.get("allergy") or
        patient.get("Allergy Details") or
        "None reported"
    )
    allergy_str = str(allergy_val).strip()
    if not allergy_str or allergy_str.lower() in ("none", "null", "n/a", ""):
        allergy_str = "None reported"
    patient_section += f"\n⚠️ ALLERGY STATUS: {allergy_str} — use this ONLY, do not infer.\n"

    vitals_section = ""
    if vitals:
        vitals_section = "\nVITALS:\n"
        for key, value in vitals.items():
            if value:
                formatted_key = key.replace("_", " ").title()
                vitals_section += f"- {formatted_key}: {value}\n"

    labs_section = ""
    if labs:
        labs_section = "\nLABORATORY RESULTS:\n"
        for key, value in labs.items():
            if value:
                formatted_key = key.replace("_", " ").title()
                labs_section += f"- {formatted_key}: {value}\n"

    diagnosis_section = ""
    if diagnosis:
        diagnosis_section = f"\nDIAGNOSIS:\n{diagnosis}\n"

    medicines_section = ""
    if medicines:
        medicines_section = "\nPRESCRIBED MEDICINES:\n"
        for med in medicines:
            if med:
                name = med.get("name", "Unknown")
                dose = med.get("dose", "")
                unit = med.get("unit", "mg")
                frequency = med.get("frequency", "")
                route = med.get("route", "")

                med_line = f"- {name}"
                if dose:
                    med_line += f" {normalize_dose_to_mg(dose, unit)}"
                if route:
                    med_line += f" ({route})"
                if frequency:
                    med_line += f" {frequency}"
                medicines_section += med_line + "\n"

    ctx_lines = []
    for item in retrieved:
        m = item.get("meta", {})
        file_path = m.get("file_path", "")
        source_label = m.get("source", "")

        if file_path:
            src = os.path.splitext(os.path.basename(file_path))[0]
        else:
            src = source_label or "UMMC Guidelines"

        page = m.get("page", "")
        ident = f"Page {int(page) + 1}" if page != "" else ""

        # Use appropriate header based on source type
        if source_label == "ANTICOAG":
            header = f"MOH Anticoagulant Quick Guide 1st Edition [{ident}]"
        else:
            header = f"UMMC Antimicrobial Guidelines 3rd Edition — Chapter {src} [{ident}]"

        snippet = item["text"].strip().replace("\n\n", "\n")
        if len(snippet) > 700:
            snippet = snippet[:700] + " ..."

        ctx_lines.append(
            f"REFERENCE [{item['rank']}]: {header}\n"
            f"{snippet}\n"
        )

    context_block = "\n".join(ctx_lines) if ctx_lines else "No context retrieved."

    # Drug relevance check section — NEW
    relevance_section = ""
    if medicines and diagnosis:
        relevance_section = (
            f"\nDRUG RELEVANCE CHECK (AUTOMATED — DO NOT OVERRIDE):\n"
            f"{check_drug_relevance(medicines, retrieved, diagnosis, per_drug_results)}\n"
        )

    # Dose check section — only for indicated drugs
    dose_check_section = ""
    if medicines:
        dose_check_section = (
            f"\nDOSE VERDICT (PYTHON-CALCULATED — DO NOT OVERRIDE):\n"
            f"{check_dose_in_range(medicines, retrieved)}\n"
        )

    prompt = (
        f"{SYSTEM_PRIMER}\n"
        f"{'=' * 70}\n"
        f"PATIENT CASE TO VERIFY\n"
        f"{'=' * 70}\n"
        f"{patient_section}"
        f"{vitals_section}"
        f"{labs_section}"
        f"{diagnosis_section}"
        f"{medicines_section}"
        f"{relevance_section}"
        f"{dose_check_section}\n"
        f"{'=' * 70}\n"
        f"RELEVANT CLINICAL EVIDENCE FROM GUIDELINES:\n"
        f"{'=' * 70}\n"
        f"NOTE: Each reference below is a specific page. "
        f"Only cite the exact page number shown — never combine into a range.\n"
        f"{'=' * 70}\n"
        f"{context_block}\n\n"
        f"{'=' * 70}\n"
        f"YOUR ANALYSIS (Do NOT repeat the format. Provide actual findings):\n"
        f"{'=' * 70}\n"
    )

    return prompt


_RE_VERIF = re.compile(
    r"^\s*(?:\*{1,2}\s*)?verification(?:\s*\*{1,2})?\s*:\s*(.+)$",
    re.I | re.M,
)


def normalize_labels(text: str) -> str:
    return re.sub(
        r"^\s*\*{1,2}\s*(Verification|Guideline Used|Confidence Score|Explanation|Citation)\s*\*{1,2}\s*:",
        r"\1:",
        text,
        flags=re.I | re.M,
    )


def strip_llm_references(text: str) -> str:
    text = re.sub(r'\n*References\s*:\s*\n.*$', '', text, flags=re.I | re.S)
    text = re.sub(r'\n*Note\s*:.*$', '', text, flags=re.I | re.S)
    text = re.sub(r'\n*Please note\s*:.*$', '', text, flags=re.I | re.S)
    return text.strip()


def ensure_verification_line(text: str) -> str:
    text = normalize_labels(text)
    text = strip_llm_references(text)

    lines = text.splitlines()
    verif_idxs = [
        i for i, line in enumerate(lines)
        if re.match(r"^\s*Verification\s*:", line, re.I)
    ]

    if len(verif_idxs) > 1:
        keep = verif_idxs[-1]
        lines = [
            line for i, line in enumerate(lines)
            if i == keep or i not in verif_idxs
        ]
        text = "\n".join(lines)

    if not _RE_VERIF.search(text):
        text = text + (
            "\nVerification: AI-generated analysis based on UMMC Antimicrobial "
            "Guidelines 3rd Edition and MOH Anticoagulant Quick Guide 1st Edition. "
            "Clinical judgment required before acting "
            "on these recommendations."
        )

    return text