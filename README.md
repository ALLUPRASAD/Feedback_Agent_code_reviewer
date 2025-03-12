
## Feedback Agent Code Reviewer

This project implements a feedback agent that reviews and improves Python code, particularly focusing on data transformation tasks involving Pandas DataFrames. It leverages LangChain and LangGraph to create a ReAct agent powered by OpenAI's GPT-4 model, enabling it to debug code, execute it in a safe environment, and process user feedback.

## Features

* **Code Debugging and Improvement:** Uses GPT-4 to analyze and enhance Python code, ensuring it defines a `transform_data` function for DataFrame manipulation.
* **Safe Code Execution:** Executes user-provided code in a restricted environment to prevent security vulnerabilities.
* **File Upload and Analysis:** Allows users to upload CSV or Excel files and process them using their custom code.
* **Feedback Processing:** Enables users to provide feedback on the code analysis and execution results, which the agent then uses to refine its output.
* **Asynchronous Processing:** Utilizes FastAPI and asynchronous operations for efficient handling of requests.
* **Dockerized Application:** Provides a Dockerfile for easy deployment and containerization.

## Architecture

* **FastAPI:** Serves as the web framework for handling API requests and responses.
* **LangChain:** Provides the interface for interacting with the OpenAI GPT-4 model and defining tools.
* **LangGraph:** Implements a ReAct agent for complex workflow management and stateful interactions.
* **Pandas:** Used for data manipulation and processing.
* **Uvicorn:** An ASGI server for running the FastAPI application.
* **Docker:** Containerizes the application for easy deployment.

## code structure

                    Feedback_Agent_code_reviewer/
                    ├── data/
                    │   ├── outputs/
                    │   └── uploads/
                    ├── src/
                    │   ├── model/
                    │   │   └── model.py
                    │   ├── tools/
                    │   │   └── tools.py
                    │   ├── validate/
                    │   │   └── validate.py
                    │   └── requirements.txt
                    ├── fastapi_app.py
                    ├── Dockerfile
                    └── README.md
                    
### Prerequisites

* Python 3.11+
* Docker (for containerized deployment)
* OpenAI API key

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/ALLUPRASAD/Feedback_Agent_code_reviewer.git
    cd Feedback_Agent_code_reviewer
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**

    * Create a `.env` file or set the `OPENAI_API_KEY` environment variable.

5.  **Run the application (local):**

    ```bash
    uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
    ```

### Docker Deployment

1.  **Build the Docker image:**

    ```bash
    docker build -t feedback-agent .
    ```

2.  **Run the Docker container:**

    ```bash
    docker run -p 8000:8000 -e OPENAI_API_KEY="YOUR_OPENAI_API_KEY" feedback-agent
    ```

    * Replace `YOUR_OPENAI_API_KEY` with your actual OpenAI API key.

### API Endpoints

* **`/upload/` (POST):** Upload a CSV or Excel file.
* **`/analyze/` (POST):** Analyze Python code and process the uploaded file.
* **`/feedback/` (POST):** Provide feedback on the analysis results.
* **`/errors/{request_id}` (GET):** Retrieve error logs.

### Example Usage


1.  **Upload a file:**

    ```bash
    curl -X POST -F "file=@your_file.csv" [http://0.0.0.0:8000/upload/](http://0.0.0.0:8000/upload/)
    ```
     **Example Response:**
          ```json
         {"request_id":"e0dc5e67-e247-41b2-8382-d7744bd2f34e","file_path":"data/uploads/e0dc5e67-e247-41b2-8382-d7744bd2f34e.csv"}
          ```
    


2.  **Analyze code:**

    ```bash
    curl -X POST [http://0.0.0.0:8000/analyze/](http://0.0.0.0:8000/analyze/) -F "user_code=@your_code.py" -F "file_path=data/uploads/your_file_uuid.csv"
    ```
    **Example Response:**
            ```json
           {"request_id":"4fb377fa-8bd3-4f83-9759-30fde00dcb42","improved_code":"The provided code is technically correct and will work without any issues if the DataFrame 'df' contains only numerical data. However, if the DataFrame contains non-numerical data (like strings or dates), this function will either fail or produce unexpected results.\n\nHere's a more robust version of the function that checks if the data is numeric before attempting to multiply it:\n\n```python\nimport pandas as pd\nimport numpy as np\n\ndef transform_data(df):\n    df = df.copy()  # create a copy of the input DataFrame to avoid modifying the original one\n    for col in df.columns:\n        if np.issubdtype(df[col].dtype, np.number):  # check if the column data type is numeric\n            df[col] = df[col] * 2  # multiply only numeric columns\n    return df\n```\n\nThis function will work correctly with DataFrames that contain both numerical and non-numerical data. It will only transform the numerical data and leave the non-numerical data unchanged.\n\nAlso, it's a good practice to add some error handling and input validation to your function. For example, you could add a check at the beginning of the function to ensure that the input is indeed a pandas DataFrame:\n\n```python\ndef transform_data(df):\n    if not isinstance(df, pd.DataFrame):\n        raise ValueError(\"Input should be a pandas DataFrame\")\n    df = df.copy()\n    for col in df.columns:\n        if np.issubdtype(df[col].dtype, np.number):\n            df[col] = df[col] * 2\n    return df\n```\n\nThis will make your function more robust and easier to use correctly.","output_file":null,"status":"error"}
            ```

3.  **Provide feedback:**

    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"request_id": "your_request_id", "feedback": "Your feedback message"}' [http://0.0.0.0:8000/feedback/](http://0.0.0.0:8000/feedback/)
    ```

    **Example Response:**
        ```json
        {"status":"success","message":"Sure, I can help with that. However, I need the code that you want me to analyze. Could you please provide it?"}
        ```

4.  **Get Error logs:**

    ```bash
    curl [http://0.0.0.0:8000/errors/your_request_id](http://0.0.0.0:8000/errors/your_request_id)
    ```
  **Example Response:**
          ```json
          {"request_id":"4751ad2c-9438-4010-a86a-d858a29524f2","log":"No logs found."}
          ```
