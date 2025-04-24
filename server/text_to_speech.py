import os
from typing import Generator
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import logging

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
logger = logging.getLogger(__name__)

def text_to_speech_stream(text: str) -> Generator[bytes, None, None]:
    try:
        response = client.text_to_speech.convert(
            voice_id="pFZP5JQG7iQjIQuC4Bku",  # Adam pre-made voice
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
                speed=1.0,
            ),
        )
        for chunk in response:
            yield chunk
    except Exception as e:
        logger.error(f"Error in text_to_speech_stream: {e}", exc_info=True)
        raise