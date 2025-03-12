# tools.py
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from src.model.model import openai_model
import pandas as pd
import os
import uuid
import logging
import timeout_decorator
from src.validate.validate import validate_code_safety, OUTPUT_DIR, EXECUTION_TIMEOUT, store_feedback #import store_feedback



# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define tools for the agent
@tool
def llm_debug_code(user_code: str) -> str:
    """Pass user code to LLM for debugging and improvements."""
    try:
        if not validate_code_safety(user_code):
            return "Error: Code contains unsafe operations or syntax errors"
        
        messages = [
            SystemMessage(content="You are a Python expert. Fix issues in the provided code and ensure it defines a 'transform_data' function that takes a DataFrame and returns a DataFrame. Suggest improvements."),
            HumanMessage(content=f"Code:\n{user_code}")
        ]
        response = openai_model.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Error in llm_debug_code: {str(e)}")
        return f"Error debugging code: {str(e)}"

@tool
@timeout_decorator.timeout(EXECUTION_TIMEOUT, timeout_exception=TimeoutError)
def execute_python_logic(code: str, input_file: str) -> str:
    """Execute Python logic in a restricted environment."""
    try:
        if not validate_code_safety(code):
            return "Error: Code contains unsafe operations or syntax errors"

        # Restricted globals
        safe_globals = {
            "__builtins__": {
                "len": len,
                "range": range,
                "str": str,
                "int": int,
                "float": float,
            },
            "pd": pd
        }
        safe_locals = {}
        
        exec(code, safe_globals, safe_locals)
        transform_func = safe_locals.get("transform_data")
        
        if not callable(transform_func):
            return "Error: No 'transform_data' function found in code"
        
        if not os.path.exists(input_file):
            return "Error: Input file not found"
        
        df = pd.read_csv(input_file) if input_file.endswith('.csv') else pd.read_excel(input_file)
        result_df = transform_func(df)
        
        if not isinstance(result_df, pd.DataFrame):
            return "Error: transform_data must return a pandas DataFrame"
        
        output_file = os.path.join(OUTPUT_DIR, f"output_{uuid.uuid4()}.xlsx")
        result_df.to_excel(output_file, index=False)
        return output_file
    except TimeoutError:
        return f"Error: Execution timed out after {EXECUTION_TIMEOUT} seconds"
    except Exception as e:
        logger.error(f"Error in execute_python_logic: {str(e)}")
        return f"Error executing code: {str(e)}"


@tool
def feedback_handler(feedback: str, request_id: str, user_code: str) -> str:
    """Store user feedback."""
    try:
        store_feedback(request_id, feedback, user_code) #use function
        logger.info(f"Feedback stored for request {request_id}")
        return "Feedback recorded successfully"
    except Exception as e:
        logger.error(f"Error in feedback_handler: {str(e)}")
        return f"Error processing feedback: {str(e)}"

