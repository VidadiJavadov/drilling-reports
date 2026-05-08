def extract_summaries(full_text: str) -> dict:
    """
    FIX: pdfplumber collapses multi-column pages into a single long line,
    so line-by-line extraction fails. Instead we use regex directly on the
    single-line text with precise anchors.

    Pattern:
      "Summary of activities (24 Hours)"  → capture until next section header
      "Summary of planned activities ..."  → capture until next section header

    The STOP anchor is any known section keyword that marks the next block.
    We also strip any trailing numeric/time garbage that may follow.
    """
    # These words mark the end of a summary block
    _SUMMARY_STOP = (
        r"(?=Summary\s+of\s+planned\s+activities|"
        r"Operations\b|"
        r"Drilling\s+Fluid\b|"
        r"Equipment\b|"
        r"Survey\s+Station\b|"
        r"Pore\s+Pressure\b|"
        r"Lithology\b|"
        r"Gas\s+Reading\b)"
    )

    act_m = re.search(
        r"Summary\s+of\s+activities[^\n]*?"
        r"\s*(.*?)"
        + _SUMMARY_STOP,
        full_text,
        re.IGNORECASE | re.DOTALL,
    )

    plan_m = re.search(
        r"Summary\s+of\s+planned\s+activities[^\n]*?"
        r"\s*(.*?)"
        + _SUMMARY_STOP,
        full_text,
        re.IGNORECASE | re.DOTALL,
    )

    def _clean_summary(m):
        if not m:
            return None
        text = m.group(1).strip()
        # Remove leading "(24 Hours)" or similar bracket prefix
        text = re.sub(r"^\(\d+\s+Hours?\)\s*", "", text, flags=re.IGNORECASE)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Must be at least 10 chars of real content
        return text if len(text) >= 10 else None

    return {
        "Summary of activities":         _clean_summary(act_m),
        "Summary of planned activities": _clean_summary(plan_m),
    }