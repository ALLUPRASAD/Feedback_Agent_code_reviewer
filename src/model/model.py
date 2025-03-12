from langchain_openai import ChatOpenAI
import os

# Load OpenAI API Key from Environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key is missing. Set OPENAI_API_KEY as an environment variable.")
openai_model = ChatOpenAI(
    model_name="gpt-4",
    openai_api_key=openai_api_key,
    temperature=0
)