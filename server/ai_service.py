import httpx
from config import GEMINI_API_KEY, SYSTEM_PROMPT, REMINDER_PROMPT
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

async def get_ai_response(message: str, is_reminder: bool = False) -> str:
    prompt = f"{REMINDER_PROMPT}{message}" if is_reminder else f"{SYSTEM_PROMPT}\n\n{message}"
    URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash-001:generateContent?key={GEMINI_API_KEY}"

    data = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 1.0,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 50 if is_reminder else 2331
        }
    }

    timeout = httpx.Timeout(30.0)  # 🔥 Set the timeout to 30 seconds

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(URL, json=data)
            response.raise_for_status()
            response_json = response.json()
            logger.debug(f"Gemini API response: {response_json}")
            return response_json["candidates"][0]["content"]["parts"][0]["text"]
        except httpx.HTTPStatusError as e:
            logger.error(f"Gemini API error: {e.response.status_code} - {e.response.text}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"API Error: {e.response.status_code}")
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing Gemini response: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error parsing API response")
        except httpx.RequestError as e:
            logger.error(f"HTTP request error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Request timeout or network error")
