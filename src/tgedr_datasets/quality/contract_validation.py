"""Validation of pandas DataFrames against ODCS data contracts.

This module loads the Great Expectations expectation suite embedded in a data
contract (``customProperties`` entry with ``property: expectations``) and
validates a pandas DataFrame against it using
:class:`tgedr_dataops.quality.pandas_validation.PandasValidation`.
"""

import logging
from pathlib import Path

import pandas as pd
import yaml
from tgedr_dataops.quality.pandas_validation import PandasValidation
from tgedr_dataops_abs.great_expectations_validation import ValidationError

logger = logging.getLogger(__name__)

_EXPECTATIONS_PROPERTY = "expectations"


def load_expectations(contract_path: Path | str) -> dict:
    """Load the Great Expectations suite embedded in an ODCS data contract.

    Parameters
    ----------
    contract_path : Path | str
        Path to the ``*.odcs.yaml`` data contract file.

    Returns
    -------
    dict
        The Great Expectations suite dict with ``expectation_suite_name`` and
        ``expectations`` keys.

    Raises
    ------
    ValidationError
        If the contract file cannot be read or contains no expectations.
    """
    logger.info("[load_expectations|in] (%s)", contract_path)

    try:
        with Path(contract_path).open(encoding="utf-8") as fd:
            contract = yaml.safe_load(fd)
    except Exception as x:
        msg = f"[load_expectations] could not read data contract '{contract_path}': {x}"
        raise ValidationError(msg) from x

    custom_properties = contract.get("customProperties") or []
    for custom_property in custom_properties:
        if custom_property.get("property") == _EXPECTATIONS_PROPERTY:
            expectations = custom_property.get("value")
            if not isinstance(expectations, dict) or not expectations.get("expectations"):
                msg = f"[load_expectations] malformed expectations in data contract '{contract_path}'"
                raise ValidationError(msg)
            logger.info("[load_expectations|out] => %d expectations", len(expectations["expectations"]))
            return expectations

    msg = f"[load_expectations] no expectations found in data contract '{contract_path}'"
    raise ValidationError(msg)


def validate_df_against_contract(df: pd.DataFrame, contract_path: Path | str) -> dict:
    """Validate a pandas DataFrame against the expectations in a data contract.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    contract_path : Path | str
        Path to the ``*.odcs.yaml`` data contract file.

    Returns
    -------
    dict
        The Great Expectations validation result dict.

    Raises
    ------
    ValidationError
        If the DataFrame does not meet the contract expectations.
    """
    logger.info("[validate_df_against_contract|in] (%s)", contract_path)

    expectations = load_expectations(contract_path)
    result = PandasValidation().validate(df, expectations)

    if not result.get("success"):
        failed = [r["expectation_config"]["type"] for r in result.get("results", []) if not r.get("success")]
        msg = f"[validate_df_against_contract] data does not meet contract '{contract_path}' expectations: {failed}"
        raise ValidationError(msg)

    logger.info("[validate_df_against_contract|out] => success")
    return result
