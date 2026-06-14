"""
Disease information service.
"""

import json
from pathlib import Path


DISEASE_INFO_PATH = (
    Path(__file__).parent.parent
    / "disease_data"
    / "disease_info.json"
)


def get_disease_info(disease_name):
    """
    Return disease advisory information.
    """

    with open(
        DISEASE_INFO_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        disease_data = json.load(file)

    return disease_data.get(
        disease_name,
        {
            "symptoms": "Information not available.",
            "treatment": "Information not available.",
            "prevention": "Information not available.",
        },
    )