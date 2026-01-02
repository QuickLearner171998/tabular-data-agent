"""Serialization utilities for converting numpy types to native Python types."""

import numpy as np
import pandas as pd
from typing import Any


def convert_to_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for JSON/msgpack serialization.
    
    Args:
        obj: Any object that may contain numpy types.
    
    Returns:
        Object with all numpy types converted to native Python types.
    """
    if obj is None:
        return None
    
    # Handle numpy arrays FIRST (before pd.isna check which fails on arrays)
    if isinstance(obj, np.ndarray):
        return [convert_to_serializable(item) for item in obj.tolist()]
    
    # Handle numpy scalar types
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, np.bool_):
        return bool(obj)
    
    # Handle pandas types
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if isinstance(obj, pd.Series):
        return convert_to_serializable(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return convert_to_serializable(obj.to_dict("records"))
    
    # Handle pandas NA (scalar only, after array checks)
    try:
        if pd.isna(obj):
            return None
    except (ValueError, TypeError):
        # pd.isna fails on certain types - that's okay
        pass
    
    # Handle containers
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    
    # Handle special float values
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    
    # Handle basic types
    if isinstance(obj, (str, int, bool)):
        return obj
    
    # Fallback: convert to string
    try:
        return str(obj)
    except Exception:
        return None
