from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from database import users_collection, chats_collection, reminders_collection, startup_event as db_startup
from auth import get_current_user, verify_token
from bson import ObjectId
from datetime import datetime
from typing import List, Optional
from ai_service import get_ai_response
from reminder_scheduler import schedule_reminder_for
from text_to_speech import text_to_speech_stream
import asyncio
import uuid
from pathlib import Path
import logging
import json
from fastapi.security import OAuth2PasswordBearer
import speech_recognition as sr
from fastapi.websockets import WebSocketState
from websocket_manager import active_connections, broadcast_reminder
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# FastAPI app setup
app = FastAPI(title="Nuera AI Chat Server", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None

class ChatResponseModel(BaseModel):
    response: str
    audio_url: str
    chat_id: str

class CreateReminder(BaseModel):
    text: str
    date: datetime

class UpdateReminder(BaseModel):
    completed: bool

class ChatResponse(BaseModel):
    id: str
    messages: List[dict]
    timestamp: datetime

class ReminderResponse(BaseModel):
    id: str
    text: str
    date: datetime
    completed: bool

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error at {request.url}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred."})

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    await db_startup()
    logger.info("Application started successfully.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down.")

# Utility function to save audio
def save_audio_to_file(text: str, audio_path: Path):
    with open(audio_path, "wb") as f:
        for chunk in text_to_speech_stream(text):
            f.write(chunk)

# Helper function for background audio generation
async def generate_and_send_audio(websocket: WebSocket, chat_id: str, message_id: str, text: str, audio_path: Path):
    try:
        # Generate audio in a separate thread
        await asyncio.to_thread(save_audio_to_file, text, audio_path)
        audio_filename = audio_path.name
        audio_url = f"/static/audio/{audio_filename}"
        
        # Update the database with the audio filename
        await chats_collection.update_one(
            {"_id": chat_id, "messages.message_id": message_id},
            {"$set": {"messages.$.audio_filename": audio_filename}}
        )
        
        # Send audio_ready message if WebSocket is still connected
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({
                "type": "audio_ready",
                "message_id": message_id,
                "audio_url": audio_url
            })
        else:
            logger.info(f"WebSocket disconnected, cannot send audio_ready for message {message_id}")
    except Exception as e:
        logger.error(f"Error generating audio for message {message_id}: {e}", exc_info=True)

# WebSocket endpoint for reminders
@app.websocket("/ws/reminders")
async def websocket_reminders(websocket: WebSocket, token: str):
    payload = verify_token(token)
    if not payload or not payload.get("id"):
        await websocket.close(code=1008)
        return
    user_id = payload["id"]
    await websocket.accept()
    active_connections.append((user_id, websocket))
    logger.info(f"WebSocket reminder connected for user: {user_id}")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if (user_id, websocket) in active_connections:
            active_connections.remove((user_id, websocket))
        logger.info("WebSocket reminder disconnected")
    except Exception as e:
        logger.error(f"WebSocket reminder error: {e}", exc_info=True)
        if (user_id, websocket) in active_connections:
            active_connections.remove((user_id, websocket))

# WebSocket endpoint for chat
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket, token: str):
    payload = verify_token(token)
    if not payload or not (user_id := payload.get("id")):
        await websocket.close(code=1008)
        return
    user = await users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    logger.info(f"WebSocket chat connected for user: {user_id}")
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            message = message_data["message"]
            chat_id = message_data.get("chat_id")
            temp_id = message_data.get("tempId")

            if not chat_id:
                chat_id = str(uuid.uuid4())
                await chats_collection.insert_one({
                    "_id": chat_id,
                    "user_id": user_id,
                    "messages": [],
                    "created_at": datetime.now(IST)
                })
                logger.info(f"New chat created: {chat_id} for user: {user_id}")

            # Store user message
            user_message_id = str(uuid.uuid4())
            user_message = {
                "message_id": user_message_id,
                "role": "user",
                "parts": [{"text": message}],
                "timestamp": datetime.now(IST),
                "audio_filename": None
            }
            result = await chats_collection.update_one(
                {"_id": chat_id, "user_id": user_id},
                {"$push": {"messages": {"$each": [user_message], "$slice": -50}}}
            )
            if result.matched_count == 0:
                logger.error(f"Failed to update chat document for chat_id: {chat_id}, user_id: {user_id}")
                await websocket.send_json({"type": "error", "content": "Chat session not found"})
                continue

            # Generate AI response
            ai_response = await get_ai_response(message)
            assistant_message_id = str(uuid.uuid4())

            # Store assistant message without audio initially
            assistant_message = {
                "message_id": assistant_message_id,
                "role": "assistant",
                "parts": [{"text": ai_response}],
                "timestamp": datetime.now(IST),
                "audio_filename": None  # Audio will be added later
            }
            result = await chats_collection.update_one(
                {"_id": chat_id, "user_id": user_id},
                {"$push": {"messages": {"$each": [assistant_message], "$slice": -50}}}
            )
            if result.matched_count == 0:
                logger.error(f"Failed to update chat document for chat_id: {chat_id}, user_id: {user_id}")
                await websocket.send_json({"type": "error", "content": "Chat update failed"})
                continue

            # Send immediate text response
            response_data = {
                "type": "text_response",
                "content": ai_response,
                "message_id": assistant_message_id,
                "chat_id": chat_id
            }
            if temp_id:
                response_data["tempId"] = temp_id
            await websocket.send_json(response_data)

            # Start background task for audio generation
            audio_filename = f"{uuid.uuid4()}.mp3"
            audio_path = Path("static/audio") / audio_filename
            asyncio.create_task(generate_and_send_audio(websocket, chat_id, assistant_message_id, ai_response, audio_path))

    except WebSocketDisconnect:
        logger.info("WebSocket chat disconnected")
    except Exception as e:
        logger.error(f"WebSocket chat error: {e}", exc_info=True)
        await websocket.send_json({"type": "error", "content": str(e)})

# WebSocket endpoint for transcription
@app.websocket("/ws/transcription")
async def websocket_transcription(websocket: WebSocket, token: str):
    payload = verify_token(token)
    if not payload or not payload.get("id"):
        await websocket.close(code=1008)
        return
    user_id = payload["id"]
    await websocket.accept()
    logger.info(f"WebSocket transcription connected for user: {user_id}")

    recognizer = sr.Recognizer()
    audio_buffer = bytearray()

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            elif message["type"] == "websocket.receive":
                if "bytes" in message:
                    audio_buffer.extend(message["bytes"])
                elif "text" in message and message["text"] == "EOS":
                    if audio_buffer:
                        audio_data = sr.AudioData(audio_buffer, sample_rate=16000, sample_width=2)
                        try:
                            transcript = recognizer.recognize_google(audio_data)
                            
                            await websocket.send_json({
                                "transcript": transcript,
                                "audio_url": f"/static/audio/{uuid.uuid4()}.mp3",
                            })

                        except sr.UnknownValueError:
                            await websocket.send_json({"transcript": "", "error": "Could not understand audio"})

                        except sr.RequestError as e:
                            await websocket.send_json({"transcript": "", "error": f"Could not request results; {e}"})

                        except Exception as e:
                            logger.error(f"Unexpected error during transcription: {e}")
                            await websocket.send_json({"transcript": "", "error": "Transcription error"})
                        audio_buffer = bytearray()  # Reset buffer
    except WebSocketDisconnect:
        logger.info("WebSocket transcription disconnected")
    except Exception as e:
        logger.error(f"WebSocket transcription error: {e}", exc_info=True)
        try:
            await websocket.send_json({"error": str(e)})
        except:
            logger.error("Failed to send error message", exc_info=True)
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass 

# Endpoint to get audio for a message
@app.get("/audio/{message_id}")
async def get_audio(message_id: str, current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$unwind": "$messages"},
        {"$match": {"messages.message_id": message_id}},
        {"$project": {"chat_id": "$_id", "message": "$messages"}}
    ]
    result = await chats_collection.aggregate(pipeline).to_list(length=1)
    if not result:
        raise HTTPException(status_code=404, detail="Message not found")
    message = result[0]["message"]
    audio_filename = message.get("audio_filename")
    if audio_filename:
        audio_url = f"/static/audio/{audio_filename}"
        return {"audio_url": audio_url}
    else:
        text = message["parts"][0]["text"]
        audio_filename = f"{uuid.uuid4()}.mp3"
        audio_path = Path("static/audio") / audio_filename
        try:
            await asyncio.to_thread(save_audio_to_file, text, audio_path)
            await chats_collection.update_one(
                {"_id": result[0]["chat_id"], "messages.message_id": message_id},
                {"$set": {"messages.$.audio_filename": audio_filename}}
            )
            audio_url = f"/static/audio/{audio_filename}"
            return {"audio_url": audio_url}
        except Exception as e:
            logger.error(f"Error generating audio: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error generating audio")

# Other endpoints remain unchanged
@app.get("/chat", response_model=List[ChatResponse])
async def get_chats(current_user: dict = Depends(get_current_user)):
    chats = await chats_collection.find({"user_id": str(current_user["_id"])}).to_list(length=None)
    return [ChatResponse(
        id=str(chat["_id"]),
        messages=chat["messages"],
        timestamp=chat.get("created_at", datetime.now(IST))
    ) for chat in chats]

@app.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str, current_user: dict = Depends(get_current_user)):
    result = await chats_collection.delete_one({"_id": chat_id, "user_id": str(current_user["_id"])})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Chat not found or unauthorized")
    return {"message": "Chat deleted successfully"}

@app.get("/user")
async def get_user(current_user: dict = Depends(get_current_user)):
    return {"username": current_user["username"]}

@app.post("/reminders", response_model=ReminderResponse)
async def create_reminder(reminder_data: CreateReminder, current_user: dict = Depends(get_current_user)):
    reminder_dict = {
        "user_id": str(current_user["_id"]),
        "text": reminder_data.text,
        "date": reminder_data.date,
        "completed": False
    }
    result = await reminders_collection.insert_one(reminder_dict)
    reminder_id = str(result.inserted_id)

    asyncio.create_task(
        schedule_reminder_for(
            reminder_id,
            str(current_user["_id"]),
            reminder_data.text,
            reminder_data.date
        )
    )

    return {
        "id": reminder_id,
        "text": reminder_data.text,
        "date": reminder_data.date,
        "completed": False
    }

@app.get("/reminders", response_model=List[ReminderResponse])
async def get_reminders(current_user: dict = Depends(get_current_user)):
    reminders = await reminders_collection.find({"user_id": str(current_user["_id"])}).to_list(length=None)
    return [ReminderResponse(
        id=str(reminder["_id"]),
        text=reminder["text"],
        date=reminder["date"],
        completed=reminder["completed"]
    ) for reminder in reminders]

@app.put("/reminders/{reminder_id}", response_model=ReminderResponse)
async def update_reminder(reminder_id: str, update_data: UpdateReminder, current_user: dict = Depends(get_current_user)):
    result = await reminders_collection.update_one(
        {"_id": ObjectId(reminder_id), "user_id": str(current_user["_id"])},
        {"$set": {"completed": update_data.completed}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Reminder not found or unauthorized")
    reminder = await reminders_collection.find_one({"_id": ObjectId(reminder_id)})
    return ReminderResponse(
        id=str(reminder["_id"]),
        text=reminder["text"],
        date=reminder["date"],
        completed=reminder["completed"]
    )

@app.delete("/reminders/{reminder_id}")
async def delete_reminder(reminder_id: str, current_user: dict = Depends(get_current_user)):
    result = await reminders_collection.delete_one({"_id": ObjectId(reminder_id), "user_id": str(current_user["_id"])})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Reminder not found or unauthorized")
    return {"message": "Reminder deleted successfully"}