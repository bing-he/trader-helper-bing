"""Facility name normalization."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

CANONICAL_OTHER = "Other / Unknown"

ALIAS_MAP: Dict[str, str] = {
    "calcasieupasslngfeedgas": "Calcasieu Pass",
    "calcasieupass": "Calcasieu Pass",
    "cameronlngfeedgas": "Cameron",
    "cameronlng": "Cameron",
    "corpuschristilngfeedgas": "Corpus Christi",
    "corpuschristi": "Corpus Christi",
    "covepointlngfeedgas": "Cove Point",
    "covepoint": "Cove Point",
    "elbaislandlngfeedgas": "Elba Island",
    "elbaisland": "Elba Island",
    "freeportlngfeedgas": "Freeport",
    "freeportlng": "Freeport",
    "sabinepasslngfeedgas": "Sabine Pass",
    "sabinepass": "Sabine Pass",
    "goldenpasslngfeedgas": "Golden Pass",
    "goldenpass": "Golden Pass",
    "uslngexportsgoldenpasslng": "Golden Pass",
    "plaquemineslngfeedgas": "Plaquemines",
    "plaquemines": "Plaquemines",
    "uslngexportsplaquemineslng": "Plaquemines",
}


def _clean_key(raw_name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", raw_name.lower())
    return key


@dataclass
class FacilityNormalizer:
    """Normalize facility names to a canonical set."""

    alias_map: Dict[str, str] = field(default_factory=lambda: ALIAS_MAP)
    unknown_seen: set = field(default_factory=set)

    def normalize_facility_name(self, raw_name: str) -> str:
        """Normalize a raw facility name to a canonical value."""
        if not isinstance(raw_name, str) or not raw_name.strip():
            return CANONICAL_OTHER
        key = _clean_key(raw_name)
        canonical = self.alias_map.get(key, CANONICAL_OTHER)
        if canonical == CANONICAL_OTHER and key not in self.unknown_seen:
            LOGGER.warning("Unknown facility encountered; assigning to %s: %s", CANONICAL_OTHER, raw_name)
            self.unknown_seen.add(key)
        return canonical

    def normalize_frame(self, df: pd.DataFrame, source: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply normalization to a dataframe and return mapping records."""
        if "Item" not in df.columns:
            raise ValueError("Expected column 'Item' to normalize facilities.")
        df = df.copy()
        df["canonical_facility"] = df["Item"].apply(self.normalize_facility_name)

        unique_map = (
            df[["Item", "canonical_facility"]]
            .drop_duplicates()
            .rename(columns={"Item": "raw_name", "canonical_facility": "canonical_name"})
        )
        unique_map["source_file"] = source
        return df, unique_map

