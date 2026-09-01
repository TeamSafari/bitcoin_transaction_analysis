from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


def _get_text(
    element,
    tag,
    default=None
):
    child = element.find(tag)

    if child is None:
        return default

    return child.text


def _get_list(
    element,
    parent_tag,
    child_tag
):
    parent = element.find(parent_tag)

    if parent is None:
        return []

    return [
        child.text
        for child in parent.findall(child_tag)
        if child.text is not None
    ]


def load_transactions_xml(
    path: Path
) -> pd.DataFrame:
    """
    Load the actual exports/transactions.xml format.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"XML file not found: {path}"
        )

    if path.suffix.lower() != ".xml":
        raise ValueError(
            f"Expected XML file: {path}"
        )

    tree = ET.parse(path)
    root = tree.getroot()

    if root.tag != "transactions":
        raise ValueError(
            f"Expected root <transactions>, "
            f"found <{root.tag}>"
        )

    records = []

    for tx in root.findall("transaction"):

        record = {
            "txid": _get_text(
                tx,
                "txid"
            ),

            "timestamp": _get_text(
                tx,
                "timestamp"
            ),

            "total_input_sats": _get_text(
                tx,
                "total_input_sats"
            ),

            "total_output_sats": _get_text(
                tx,
                "total_output_sats"
            ),

            "fee_sats": _get_text(
                tx,
                "fee_sats"
            ),

            "script_type": _get_text(
                tx,
                "script_type"
            ),

            "input_addresses": _get_list(
                tx,
                "input_addresses",
                "address"
            ),

            "input_amounts_sats": _get_list(
                tx,
                "input_amounts_sats",
                "amount"
            ),

            "output_addresses": _get_list(
                tx,
                "output_addresses",
                "address"
            ),

            "output_amounts_sats": _get_list(
                tx,
                "output_amounts_sats",
                "amount"
            ),
        }

        records.append(record)

    return pd.DataFrame(records)