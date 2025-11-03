"""Evaluation metrics"""

from typing import List
from sklearn.metrics import accuracy_score, f1_score


def extract_output(response: str) -> str:
    """
    Extract the output label from a response string.

    Args:
        response: Model response text

    Returns:
        Extracted label (lowercase, stripped)
    """
    response_lower = response.lower().strip()
    if "output:" in response_lower:
        # Extract text after "output:"
        answer = response_lower.split("output:")[-1].strip()
        # Remove any trailing punctuation or whitespace
        return answer.split()[0] if answer else ""
    return response_lower


def calculate_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """
    Calculate accuracy score.

    Args:
        y_true: List of true labels
        y_pred: List of predicted labels

    Returns:
        Accuracy score (0.0 to 1.0)
    """
    if not y_true or not y_pred:
        return 0.0
    return float(accuracy_score(y_true, y_pred))


def calculate_f1(y_true: List[str], y_pred: List[str], average: str = "macro") -> float:
    """
    Calculate F1 score.

    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
        average: Averaging strategy ("macro", "micro", "weighted")

    Returns:
        F1 score
    """
    if not y_true or not y_pred:
        return 0.0
    return float(f1_score(y_true, y_pred, average=average))

