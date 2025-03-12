# remaining.py
import ast
from typing import Dict, Any

# Configuration
OUTPUT_DIR = "data/outputs"
EXECUTION_TIMEOUT = 30  # seconds

# Initialize storage
debug_logs: Dict[str, str] = {}
feedback_store: Dict[str, Dict[str, Any]] = {}

# Basic code validation
def validate_code_safety(code: str) -> bool:
    """Perform basic safety checks on the code."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                return False
            if isinstance(node, ast.Attribute):
                if node.attr in ("__import__", "eval", "exec", "open", "os", "sys"):
                    return False
        return True
    except SyntaxError:
        return False
def store_feedback(request_id: str, feedback: str, user_code: str):
    feedback_store[request_id] = {
        "feedback": feedback,
        "user_code": user_code,
        "timestamp": pd.Timestamp.now().isoformat()}