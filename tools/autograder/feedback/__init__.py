"""TA-review-only LLM feedback drafts for batch grading."""

from .batch_feedback import generate_feedback_for_student
from .client import FeedbackClientConfig, FeedbackResult, generate_feedback
from .prompt_builder import FeedbackPrompt, FeedbackPromptContext, build_feedback_prompt

__all__ = [
    "FeedbackClientConfig",
    "FeedbackPrompt",
    "FeedbackPromptContext",
    "FeedbackResult",
    "build_feedback_prompt",
    "generate_feedback",
    "generate_feedback_for_student",
]
