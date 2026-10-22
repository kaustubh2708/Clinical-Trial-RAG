"""Download a real trial's eligibility criteria from the ClinicalTrials.gov API v2 into a .txt file.

Usage:
    python -m scripts.fetch_ctgov NCT04368728 [--out data/raw_protocols]

The output uses the "Inclusion Criteria:" / "Exclusion Criteria:" markers the splitter understands,
so it can be uploaded directly to POST /api/v1/protocols/index.
"""

import argparse
import re
import sys
from pathlib import Path

import httpx

API_URL = "https://clinicaltrials.gov/api/v2/studies/{nct_id}"


def fetch_study(nct_id: str) -> dict:
    response = httpx.get(API_URL.format(nct_id=nct_id), params={"format": "json"}, timeout=30.0)
    response.raise_for_status()
    return response.json()


def to_protocol_text(study: dict) -> str:
    section = study["protocolSection"]
    ident = section.get("identificationModule", {})
    eligibility = section.get("eligibilityModule", {})
    nct_id = ident.get("nctId", "")
    lines = [
        "## Study Overview",
        f"Protocol ID: {nct_id}",
        f"Title: {ident.get('officialTitle') or ident.get('briefTitle', '')}",
        f"Source: https://clinicaltrials.gov/study/{nct_id}",
        "",
        "## Eligibility Summary",
        f"Sex: {eligibility.get('sex', 'n/a')}",
        f"Minimum age: {eligibility.get('minimumAge', 'n/a')}",
        f"Maximum age: {eligibility.get('maximumAge', 'n/a')}",
        f"Healthy volunteers: {eligibility.get('healthyVolunteers', 'n/a')}",
        "",
        # CT.gov text already contains "Inclusion Criteria:" / "Exclusion Criteria:" with "* " bullets.
        re.sub(r"\\([<>*\-])", r"\1", eligibility.get("eligibilityCriteria", "")),
    ]
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nct_id")
    parser.add_argument("--out", type=Path, default=Path("data/raw_protocols"))
    args = parser.parse_args()
    if not re.fullmatch(r"NCT\d{8}", args.nct_id):
        sys.exit("nct_id must look like NCT01234567")

    text = to_protocol_text(fetch_study(args.nct_id))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.nct_id}_ctgov.txt"
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path} ({len(text)} chars)")


if __name__ == "__main__":
    main()
