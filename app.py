from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import os
import shutil
import uuid
import logging
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from src.tools.tools import llm_debug_code, execute_python_logic, feedback_handler
import pandas as pd
import uvicorn
from langgraph.prebuilt import create_react_agent
import src.validate.validate  # Correct import
from src.model.model import openai_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_DIR = "data/uploads"
OUTPUT_DIR = "data/outputs"
ALLOWED_EXTENSIONS = {"csv", "xlsx"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)



app = FastAPI()


# Initialize storage (using remaining module)
debug_logs: Dict[str, str] = src.validate.validate.debug_logs
feedback_store: Dict[str, Dict[str, Any]] = src.validate.validate.feedback_store

# Initialize memory and agent
checkpointer = MemorySaver()
tools = [llm_debug_code, execute_python_logic, feedback_handler]

agent = create_react_agent(
    model=openai_model,
    tools=tools,
    checkpointer=checkpointer
)

class AnalysisResponse(BaseModel):
    request_id: str
    improved_code: Optional[str] = None
    output_file: Optional[str] = None
    status: str

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, str]:
    """Handle file uploads with security checks."""
    try:
        file_ext = file.filename.split(".")[-1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Invalid file format. Only CSV and Excel allowed.")
        
        file.file.seek(0, os.SEEK_END)
        if file.file.tell() > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")
        file.file.seek(0)
        
        request_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{request_id}.{file_ext}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File uploaded: {file_path}")
        return {"request_id": request_id, "file_path": file_path}
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")
    finally:
        await file.close()

@app.post("/analyze/", response_model=AnalysisResponse)
async def analyze_code(user_code: str = Form(...), file_path: str = Form(...)):
    """Process user code through agent workflow."""
    try:
        request_id = str(uuid.uuid4())
        thread_id = f"thread_{request_id}"
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="Input file not found")

        # Execute agent workflow
        result = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content=f"Debug this code and execute it:\n{user_code}")
                ],
                "input_file": file_path,
            },
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "analysis_namespace",
                    "checkpoint_id": f"checkpoint_{request_id}"
                }
            }
        )

        last_message = result["messages"][-1]
        improved_code = None
        output_file = None
        
        # Extract results
        for msg in reversed(result["messages"]):
            if isinstance(msg, HumanMessage):
                continue
            content = msg.content if hasattr(msg, "content") else str(msg)
            if "output_" in content and ".xlsx" in content:
                output_file = content
            elif "def transform_data" in content:
                improved_code = content
        
        status = "success" if output_file and "Error" not in output_file else "error"
        debug_logs[request_id] = output_file if status == "error" else "Execution successful"
        feedback_store[request_id] = {"user_code": user_code}

        logger.info(f"Analysis completed for request {request_id}")
        return AnalysisResponse(
            request_id=request_id,
            improved_code=improved_code,
            output_file=output_file,
            status=status
        )
    except Exception as e:
        logger.error(f"Error in analyze_code: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing code: {str(e)}")

@app.get("/errors/{request_id}")
async def get_error_logs(request_id: str):
    """Retrieve error logs."""
    log = debug_logs.get(request_id, "No logs found.")
    logger.info(f"Retrieved logs for request {request_id}")
    return {"request_id": request_id, "log": log}

class FeedbackRequest(BaseModel):
    request_id: str
    feedback: str

@app.post("/feedback/")
async def collect_feedback(request: FeedbackRequest):
    """Process user feedback."""
    try:
        thread_id = f"thread_{request.request_id}"
        original_data = feedback_store.get(request.request_id, {})
        user_code = original_data.get("user_code", "No code found")

        result = await agent.ainvoke(
            {
                "messages": [HumanMessage(content=f"Feedback: {request.feedback}")],
                "feedback": request.feedback,
                "request_id": request.request_id,
                "user_code": user_code,
            },
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "feedback_namespace",
                    "checkpoint_id": f"checkpoint_feedback_{request.request_id}"
                }
            }
        )

        logger.info(f"Feedback processed for request {request.request_id}")
        return {"status": "success", "message": result["messages"][-1].content}
    except Exception as e:
        logger.error(f"Error in collect_feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")

@app.on_event("shutdown")
async def cleanup():
    """Clean up temporary files."""
    for dir_path in (UPLOAD_DIR, OUTPUT_DIR):
        for file in os.listdir(dir_path):
            try:
                os.remove(os.path.join(dir_path, file))
                logger.info(f"Cleaned up file: {file}")
            except Exception as e:
                logger.error(f"Error cleaning up file {file}: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)