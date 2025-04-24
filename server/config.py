import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
JWT_SECRET_KEY = os.getenv("JWT_SECRET")

# MongoDB
MONGO_URI = os.getenv("MONGO_URI")

# Prompts
SYSTEM_PROMPT = (
    "You are Nuera, a friendly AI buddy. Keep responses fun, engaging, and knowledgeable."
)
REMINDER_PROMPT = "Generate a short, casual reminder message (max 10 words) for: "

# Validate required environment variables
required_vars = ["GEMINI_API_KEY", "ELEVENLABS_API_KEY", "JWT_SECRET", "MONGO_URI"]
for var in required_vars:
    if not os.getenv(var):
        raise ValueError(f"Environment variable {var} is not set")