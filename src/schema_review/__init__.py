"""Evidence-classed review and executable feature/fallback tools."""
from .fallback_runtime import FallbackError, execute_and_validate
from .feature_fallback import classify_feature, classify_model
from .validate import load_schema, validate_review

__all__ = [
    "FallbackError",
    "classify_feature",
    "classify_model",
    "execute_and_validate",
    "load_schema",
    "validate_review",
]
__version__ = "0.4.0"
