from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, Any, Optional, List
import pandas as pd
import os
import tempfile
import shutil
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
import time
from RestrictedPython import compile_restricted_exec, safe_globals
import uvicorn
from pathlib import Path
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Constants
CONFIG = {
    "TIMEOUT": 30,
    "TEMP_DIR": Path(tempfile.gettempdir()),
    "ALLOWED_EXTENSIONS": {".csv", ".xlsx"},
    "MAX_FILE_SIZE": 10 * 1024 * 1024  # 10MB limit
}

# Default Config (no-op transformation)
DEFAULT_CONFIG = {
    "initial_logic": {"code": "result = df"},
    "transformations": {}
}

# Secure Sandbox Execution
def execute_in_sandbox(code: str, data: pd.DataFrame, stage: str) -> Dict[str, Any]:
    if not isinstance(code, str):
        logger.error(f"Invalid code type in stage {stage}: {type(code)}")
        return {"success": False, "data": None, "error": f"Code must be a string, got {type(code)}"}
    
    restricted_globals = {
        "__builtins__": {
            "len": len,
            "range": range,
            "abs": abs,
            "str": str,
            "int": int,
            "float": float
        },
        "df": data.copy(),
        "pd": pd,
        "_getitem_": lambda x, y: x[y],
    }
    
    if any(keyword in code.lower() for keyword in ["import", "os", "open", "eval", "exec"]):
        logger.error(f"Unsafe code detected in stage {stage}")
        return {"success": False, "data": None, "error": "Unsafe code detected"}
    
    try:
        start_time = time.time()
        logger.info(f"Starting sandbox execution in stage {stage} with code:\n{code}")
        byte_code = compile_restricted_exec(code, filename='<inline>')
        if byte_code.errors:
            logger.error(f"Sandbox compilation errors in stage {stage}: {byte_code.errors}")
            return {"success": False, "data": None, "error": "; ".join(byte_code.errors)}
        
        exec(byte_code, restricted_globals)
        result = restricted_globals.get("result", data)
        
        if time.time() - start_time > CONFIG["TIMEOUT"]:
            logger.error(f"Execution timeout in stage {stage} after {CONFIG['TIMEOUT']} seconds")
            raise TimeoutError(f"Execution exceeded timeout of {CONFIG['TIMEOUT']} seconds")
            
        logger.info(f"Sandbox execution completed successfully in stage {stage}")
        return {"success": True, "data": result.to_dict('records'), "error": None}
    except Exception as e:
        logger.error(f"Sandbox execution failed in stage {stage}: {str(e)}\nTraceback: {traceback.format_exc()}")
        return {"success": False, "data": None, "error": str(e)}

# State Definition
class WorkflowState(BaseModel):
    file_path: str
    data: Optional[List[Dict]] = None
    errors: Optional[str] = None
    error_trace: Dict[str, str] = {}
    user_feedback: Dict[str, Any] = {}
    output_file: Optional[str] = None
    config: Dict[str, Any] = {}
    execution_count: int = 0
    stage_reports: Dict[str, Dict] = {}
    fallback_data: Optional[List[Dict]] = None

# Helper Functions
def dataframe_to_dict(df: Optional[pd.DataFrame]) -> Optional[List[Dict]]:
    return df.to_dict('records') if df is not None else None

def dict_to_dataframe(data_list: Optional[List[Dict]]) -> Optional[pd.DataFrame]:
    return pd.DataFrame(data_list) if data_list is not None else None

# Workflow Nodes
def load_data(state: WorkflowState) -> WorkflowState:
    stage = "load_data"
    logger.info(f"Starting {stage}")
    try:
        if not os.path.exists(state.file_path):
            raise FileNotFoundError(f"File not found: {state.file_path}")
        if Path(state.file_path).suffix.lower() not in CONFIG["ALLOWED_EXTENSIONS"]:
            raise ValueError("Unsupported file format")
        if os.path.getsize(state.file_path) > CONFIG["MAX_FILE_SIZE"]:
            raise ValueError("File size exceeds maximum limit")
            
        df = pd.read_csv(state.file_path) if state.file_path.endswith(".csv") else pd.read_excel(state.file_path)
        state.data = dataframe_to_dict(df)
        state.fallback_data = state.data.copy()
        state.execution_count = 1
        state.stage_reports[stage] = {"status": "completed", "data_shape": df.shape}
        logger.info(f"{stage} completed successfully")
    except Exception as e:
        state.errors = str(e)
        state.error_trace[stage] = traceback.format_exc()
        state.stage_reports[stage] = {"status": "failed", "error": str(e)}
        logger.error(f"{stage} failed: {str(e)}")
    return state

def initial_python_logic(state: WorkflowState) -> WorkflowState:
    stage = "initial_python_logic"
    logger.info(f"Starting {stage}")
    initial_logic = state.config.get("initial_logic", DEFAULT_CONFIG["initial_logic"])
    code = initial_logic.get("code", "")
    if state.errors or not code or not isinstance(code, str):
        logger.warning(f"Skipping {stage} due to existing errors or invalid/missing code: {code}")
        state.stage_reports[stage] = {"status": "skipped", "reason": f"Errors or invalid/missing code: {code}"}
        return state
    data_df = dict_to_dataframe(state.data)
    try:
        execution_result = execute_in_sandbox(code, data_df.copy(), stage)
        if not execution_result["success"]:
            state.errors = execution_result["error"]
            state.error_trace[stage] = execution_result["error"]
            state.stage_reports[stage] = {"status": "failed", "error": execution_result["error"]}
            logger.error(f"{stage} failed: {execution_result['error']}")
            return state
        state.data = execution_result["data"]
        state.execution_count += 1
        state.stage_reports[stage] = {"status": "completed", "data_shape": dict_to_dataframe(state.data).shape}
        logger.info(f"{stage} completed successfully")
    except Exception as e:
        state.errors = str(e)
        state.error_trace[stage] = traceback.format_exc()
        state.stage_reports[stage] = {"status": "failed", "error": str(e)}
        logger.error(f"{stage} failed: {str(e)}")
    return state

def openai_code_interpreter(state: WorkflowState) -> WorkflowState:
    stage = "openai_code_interpreter"
    logger.info(f"Starting {stage}")
    if state.errors:
        logger.info(f"Recovering in {stage} using fallback data due to previous errors")
        state.data = state.fallback_data
        state.errors = None
        state.stage_reports[stage] = {"status": "recovered", "reason": "Previous errors, using fallback data"}
    config = state.config.get("transformations", DEFAULT_CONFIG["transformations"])
    if not any([config.get(k) for k in ["filter", "merge", "group_by", "format"]]):
        logger.warning(f"Skipping {stage} due to missing transformations")
        state.stage_reports[stage] = {"status": "skipped", "reason": "Missing transformations"}
        return state
    data_df = dict_to_dataframe(state.data)
    try:
        code_lines = ["result = df"]
        if config.get("filter"):
            column = config["filter"]["column"]
            threshold = config["filter"]["threshold"]
            operator = config["filter"].get("operator", "gt")
            code_lines.append(f"if '{operator}' == 'gt':")
            code_lines.append(f"    result = df[df['{column}'] > {threshold}]")
            code_lines.append(f"elif '{operator}' == 'lt':")
            code_lines.append(f"    result = df[df['{column}'] < {threshold}]")
        if config.get("merge"):
            column = config["merge"]["column"]
            code_lines.append(f"dummy_df = pd.DataFrame({{'{column}': df['{column}'].unique(), 'info': [f'Info_{{i}}' for i in range(len(df['{column}'].unique()))]}})")
            code_lines.append(f"result = pd.merge(df, dummy_df, on='{column}', how='left')")
        if config.get("group_by"):
            group_by_col = config["group_by"]["column"]
            agg_cols = config["group_by"].get("agg_columns", [])
            agg_method = config["group_by"].get("agg_method", "mean")
            code_lines.append(f"result = df.groupby('{group_by_col}')[{agg_cols}].agg('{agg_method}').reset_index()")
        if config.get("format"):
            format_cols = config["format"].get("columns", [])
            code_lines.append(f"for col in {format_cols}:")
            code_lines.append("    if col in df.columns:")
            code_lines.append("        result[f'formatted_{col}'] = result[col].apply(lambda x: f'Formatted: {x:.2f}' if pd.notnull(x) and isinstance(x, (int, float)) else str(x))")
        
        code = "\n".join(code_lines)
        execution_result = execute_in_sandbox(code, data_df.copy(), stage)
        if not execution_result["success"]:
            state.errors = execution_result["error"]
            state.error_trace[stage] = execution_result["error"]
            state.stage_reports[stage] = {"status": "failed", "error": execution_result["error"]}
            logger.error(f"{stage} failed: {execution_result['error']}")
            state.data = state.fallback_data
            return state
        state.data = execution_result["data"]
        state.execution_count += 1
        state.stage_reports[stage] = {"status": "completed", "data_shape": dict_to_dataframe(state.data).shape}
        logger.info(f"{stage} completed successfully")
    except Exception as e:
        state.errors = str(e)
        state.error_trace[stage] = traceback.format_exc()
        state.stage_reports[stage] = {"status": "failed", "error": str(e)}
        state.data = state.fallback_data
        logger.error(f"{stage} failed: {str(e)}")
    return state

def openai_code_with_feedback(state: WorkflowState) -> WorkflowState:
    stage = "openai_code_with_feedback"
    logger.info(f"Starting {stage}")
    if not state.user_feedback:
        logger.warning(f"Skipping {stage} due to no feedback")
        state.stage_reports[stage] = {"status": "skipped", "reason": "No feedback provided"}
        return state
    # Reset errors to allow feedback processing even after previous failures
    if state.errors:
        logger.info(f"Recovering in {stage} due to previous errors: {state.errors}")
        state.errors = None
        state.data = state.fallback_data or state.data
    data_df = dict_to_dataframe(state.data)
    try:
        code_lines = ["result = df"]
        if "transformations" in state.config and state.user_feedback.get("adjustments"):
            config = state.config["transformations"]
            adjustments = state.user_feedback["adjustments"]
            if config.get("filter") and adjustments.get("filter"):
                column = config["filter"]["column"]
                threshold = adjustments["filter"].get("threshold", 0)  # Default to 0 if not provided
                operator = adjustments["filter"].get("operator", "gt")
                code_lines.append(f"if '{operator}' == 'gt':")
                code_lines.append(f"    result = df[df['{column}'] > {threshold}]")
                code_lines.append(f"elif '{operator}' == 'lt':")
                code_lines.append(f"    result = df[df['{column}'] < {threshold}]")
        code = "\n".join(code_lines)
        logger.info(f"Generated code for {stage}:\n{code}")
        execution_result = execute_in_sandbox(code, data_df.copy(), stage)
        if not execution_result["success"]:
            state.errors = execution_result["error"]
            state.error_trace[stage] = execution_result["error"]
            state.stage_reports[stage] = {"status": "failed", "error": execution_result["error"]}
            logger.error(f"{stage} failed: {execution_result['error']}")
            return state
        state.data = execution_result["data"]
        state.errors = None
        state.execution_count += 1
        state.stage_reports[stage] = {"status": "completed", "data_shape": dict_to_dataframe(state.data).shape}
        logger.info(f"{stage} completed successfully")
    except Exception as e:
        state.errors = str(e)
        state.error_trace[stage] = traceback.format_exc()
        state.stage_reports[stage] = {"status": "failed", "error": str(e)}
        logger.error(f"{stage} failed: {str(e)}")
    return state

def export_to_excel(state: WorkflowState) -> WorkflowState:
    stage = "export_to_excel"
    logger.info(f"Starting {stage}")
    if state.data is not None:
        data_df = dict_to_dataframe(state.data)
        output_path = CONFIG["TEMP_DIR"] / f"output_{state.execution_count}.xlsx"
        data_df.to_excel(output_path, index=False)
        state.output_file = str(output_path)
        state.stage_reports[stage] = {"status": "completed", "output_path": str(output_path)}
        logger.info(f"{stage} completed: Output saved to {output_path}")
    else:
        state.stage_reports[stage] = {"status": "failed", "error": "No data to export"}
        logger.warning(f"{stage} failed: No data to export")
    return state

# Define the Graph
workflow = StateGraph(WorkflowState)
workflow.add_node("load_data", load_data)
workflow.add_node("initial_python_logic", initial_python_logic)
workflow.add_node("openai_code_interpreter", openai_code_interpreter)
workflow.add_node("openai_code_with_feedback", openai_code_with_feedback)
workflow.add_node("export_to_excel", export_to_excel)

workflow.set_entry_point("load_data")
workflow.add_edge("load_data", "initial_python_logic")
workflow.add_edge("initial_python_logic", "openai_code_interpreter")
workflow.add_edge("openai_code_interpreter", "openai_code_with_feedback")
workflow.add_edge("openai_code_with_feedback", "export_to_excel")
workflow.add_edge("export_to_excel", END)

app_graph = workflow.compile(checkpointer=MemorySaver())

# FastAPI App
app = FastAPI(title="Generic Secure Data Transformation Service")

class ExecutionRequest(BaseModel):
    config: Dict[str, Any] = None

task_results: Dict[str, WorkflowState] = {}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), task_id: str = None):
    if task_id is None:
        task_id = f"task_{int(time.time())}"
    if Path(file.filename).suffix.lower() not in CONFIG["ALLOWED_EXTENSIONS"]:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            if os.path.getsize(tmp_file.name) > CONFIG["MAX_FILE_SIZE"]:
                os.unlink(tmp_file.name)
                raise HTTPException(status_code=400, detail="File size exceeds maximum limit")
            initial_state = WorkflowState(file_path=tmp_file.name)
            task_results[task_id] = initial_state
            logger.info(f"Uploaded file for task_id: {task_id}")
            return {"task_id": task_id, "message": "File uploaded successfully"}
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/execute/{task_id}")
async def execute_transformation(task_id: str, request: ExecutionRequest, background_tasks: BackgroundTasks):
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="Task not found")
    state = task_results[task_id]
    state.config = {**DEFAULT_CONFIG, **(request.config or {})}  # Merge default config with provided config
    
    async def run_workflow():
        try:
            result = await app_graph.ainvoke(state, config={"configurable": {"thread_id": task_id}})
            if not isinstance(result, WorkflowState):
                result = WorkflowState(**result)
            task_results[task_id] = result
            logger.info(f"Transformation completed for task_id: {task_id}")
        except Exception as e:
            state.errors = str(e)
            state.error_trace["workflow"] = traceback.format_exc()
            task_results[task_id] = state
            logger.error(f"Transformation failed for task_id: {task_id}, error: {str(e)}")
    
    background_tasks.add_task(run_workflow)
    return {"task_id": task_id, "message": "Transformation started"}

@app.post("/feedback/{task_id}")
async def submit_feedback(task_id: str, feedback: Dict[str, Any]):
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="Task not found")
    state = task_results[task_id]
    state.user_feedback.update(feedback or {})
    try:
        logger.info(f"Attempting to apply feedback for task_id: {task_id}, feedback: {state.user_feedback}")
        result = await app_graph.ainvoke(state, config={"configurable": {"thread_id": task_id}})
        if not isinstance(result, WorkflowState):
            result = WorkflowState(**result)
        task_results[task_id] = result
        logger.info(f"Feedback applied successfully for task_id: {task_id}")
        return {"task_id": task_id, "message": "Feedback applied", "status": "success"}
    except Exception as e:
        logger.error(f"Feedback failed for task_id: {task_id}, error: {str(e)}\nTraceback: {traceback.format_exc()}")
        state.errors = str(e)
        state.error_trace["feedback"] = traceback.format_exc()
        task_results[task_id] = state
        raise HTTPException(status_code=500, detail=f"Feedback failed: {str(e)}")

@app.get("/results/{task_id}")
async def get_results(task_id: str):
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="Task not found")
    state = task_results[task_id]
    logger.info(f"Retrieving results for task_id: {task_id}")
    if not isinstance(state, WorkflowState):
        logger.error(f"Invalid state object for task_id: {task_id}")
        raise HTTPException(status_code=500, detail="Invalid state object")
    
    response = {
        "task_id": task_id,
        "status": "completed" if state.errors is None else "failed",
        "data": state.data,
        "errors": state.errors,
        "error_trace": state.error_trace,
        "execution_count": state.execution_count,
        "stage_reports": state.stage_reports
    }
    
    if state.output_file and os.path.exists(state.output_file):
        response["download_url"] = f"/download/{task_id}"
    
    return JSONResponse(content=response)

@app.get("/download/{task_id}")
async def download_result(task_id: str):
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="Task not found")
    state = task_results[task_id]
    if not state.output_file or not os.path.exists(state.output_file):
        raise HTTPException(status_code=404, detail="Output file not available")
    logger.info(f"Download initiated for task_id: {task_id}")
    return FileResponse(state.output_file, filename="transformed_data.xlsx")

@app.get("/errors/{task_id}")
async def get_errors(task_id: str):
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="Task not found")
    state = task_results[task_id]
    logger.info(f"Retrieving errors for task_id: {task_id}")
    return {
        "task_id": task_id,
        "errors": state.errors,
        "error_trace": state.error_trace,
        "stage_reports": state.stage_reports
    }



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)