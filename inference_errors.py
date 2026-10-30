"""Inference failures are distinct from per-sample context limits."""

class ContextLimitError(ValueError):
    """The actual rendered prompt/generation budget exceeds a configured limit."""


class StructuredOutputError(RuntimeError):
    """Schema/backend configuration error: abort the experiment, never score as a case."""


class InferenceBackendError(RuntimeError):
    """Engine failure: abort rather than manufacture failed task predictions."""
