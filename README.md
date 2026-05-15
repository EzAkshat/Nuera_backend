<div align="center">

<br />

# 🤖 Nuera AI Chat Server

**A real-time AI chat backend built with FastAPI & Python.**  
Supports WebSocket-based AI conversations, voice I/O, smart reminders, and JWT-secured sessions — all wired to MongoDB.

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=13&pause=2000&color=6B7280&center=true&vCenter=true&repeat=true&width=480&lines=Real-Time+AI+Chat+via+WebSocket;Text-to-Speech+with+ElevenLabs;Speech-to-Text+Transcription;Smart+Reminder+Scheduling;JWT+Authentication+%2B+MongoDB" alt="Typing SVG" />

[Features](#-features) · [Tech Stack](#-tech-stack) · [Getting Started](#-getting-started) · [API Reference](#-api-reference) · [Message Flow](#-message-flow) · [Folder Structure](#-folder-structure) · [Security](#️-security)

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-000?style=flat-square&logo=python&logoColor=4ade80&labelColor=111" />
  <img src="https://img.shields.io/badge/FastAPI-0.104+-000?style=flat-square&logo=fastapi&logoColor=fff&labelColor=111" />
  <img src="https://img.shields.io/badge/MongoDB-Motor-000?style=flat-square&logo=mongodb&logoColor=4ade80&labelColor=111" />
  <img src="https://img.shields.io/badge/Gemini-2.0_Flash-000?style=flat-square&logo=google&logoColor=fff&labelColor=111" />
  <img src="https://img.shields.io/badge/ElevenLabs-TTS-000?style=flat-square&logoColor=fff&labelColor=111" />
</p>

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 **AI Chat** | Real-time conversations with Gemini 2.0 Flash over WebSocket |
| 🔊 **Text-to-Speech** | AI responses streamed as MP3 audio via ElevenLabs |
| 🎙️ **Speech-to-Text** | Live voice input transcription over WebSocket using Google STT |
| ⏰ **Smart Reminders** | Schedule reminders that fire AI-generated follow-up messages at the set time |
| 📡 **Live Updates** | WebSocket channels for both chat and reminder broadcast |
| 🔑 **JWT Auth** | Stateless token-based authentication on all protected routes |
| 🗄️ **Persistent Storage** | Full chat history and reminders stored in MongoDB |
| 🕐 **IST Timezone** | All reminders scheduled and delivered in Asia/Kolkata time |
| 📋 **Structured Logging** | Request and error logging via Python's logging module |

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Runtime** | Python 3.10+ |
| **Framework** | FastAPI + Uvicorn |
| **Database** | MongoDB via Motor (async) |
| **AI Model** | Google Gemini 2.0 Flash |
| **Text-to-Speech** | ElevenLabs |
| **Speech-to-Text** | SpeechRecognition (Google) |
| **Auth** | JWT via python-jose |
| **Password Hashing** | passlib + bcrypt |
| **HTTP Client** | HTTPX (async) |
| **Config** | python-dotenv |

---

## 🚀 Getting Started

### Prerequisites

- [Python](https://www.python.org/) 3.10+
- [MongoDB](https://www.mongodb.com/) (local or Atlas)
- A [Google Gemini](https://ai.google.dev/) API key
- An [ElevenLabs](https://elevenlabs.io/) API key

### 1. Clone the repository

```bash
git clone https://github.com/EzAkshat/Nuera_backend.git
cd Nuera_backend/server
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the `server/` directory:

```env
GEMINI_API_KEY=your_gemini_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
JWT_SECRET=your_jwt_secret_key
MONGO_URI=mongodb://localhost:27017
```

### 5. Create the audio directory

```bash
mkdir -p static/audio
```

### 6. Start the server

```bash
uvicorn main:app --reload --port 8000
```

The server will be live at **`http://localhost:8000`**  
Interactive API docs at **`http://localhost:8000/docs`**

---

## 📁 Folder Structure

```txt
Nuera_backend/
│
├── server/
│   ├── main.py                  # App entry point — all routes & WebSocket handlers
│   ├── ai_service.py            # Google Gemini API integration
│   ├── auth.py                  # JWT verification & current user dependency
│   ├── config.py                # Environment variables & AI prompts
│   ├── database.py              # MongoDB client & collection references
│   ├── reminder_scheduler.py    # Async reminder scheduling logic
│   ├── text_to_speech.py        # ElevenLabs TTS streaming
│   ├── websocket_manager.py     # Active WebSocket connection registry
│   └── static/
│       └── audio/               # Generated MP3 audio files
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🔌 API Reference

### HTTP Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/login` | ❌ | Obtain a JWT token |
| `GET` | `/user` | ✅ | Get current user info |
| `GET` | `/chat` | ✅ | Fetch all chat sessions |
| `DELETE` | `/chat/{chat_id}` | ✅ | Delete a chat session |
| `POST` | `/reminders` | ✅ | Create and schedule a reminder |
| `GET` | `/reminders` | ✅ | List all reminders |
| `PUT` | `/reminders/{id}` | ✅ | Update reminder status |
| `DELETE` | `/reminders/{id}` | ✅ | Delete a reminder |
| `GET` | `/audio/{message_id}` | ✅ | Fetch TTS audio for a message |

### WebSocket Endpoints

| Endpoint | Description |
|---|---|
| `/ws/chat?token=` | Real-time AI chat — send messages, receive text + audio events |
| `/ws/reminders?token=` | Live reminder notifications when a scheduled reminder fires |
| `/ws/transcription?token=` | Stream audio bytes for speech-to-text transcription |

---

## 🔄 Message Flow

### 💬 WebSocket Chat

```
Client connects to /ws/chat?token=jwt
        │
        ▼
Server verifies JWT ──► Reject (code 1008) if invalid
        │
        ▼
Client sends JSON ──► { "message": "Hey!", "chat_id": null }
        │
        ▼
New chat_id created (if none) ──► User message saved to MongoDB
        │
        ▼
POST to Gemini API ──► AI response returned
        │
        ▼
{ type: "text_response", content: "...", message_id, chat_id }
        │
        ▼
Background task: ElevenLabs TTS ──► MP3 saved to static/audio/
        │
        ▼
{ type: "audio_ready", message_id, audio_url } ──► sent when ready
```

### ⏰ Reminder Flow

```
POST /reminders ──► Validate input ──► Save to MongoDB
        │
        ▼
asyncio.create_task(schedule_reminder_for(...))
        │
        ▼
Scheduler sleeps until reminder datetime (IST)
        │
        ▼
Gemini generates short follow-up message
        │
        ▼
Reminder marked completed in DB
        │
        ▼
broadcast_reminder() ──► WebSocket push to user
```

### 🎙️ Transcription Flow

```
Client connects to /ws/transcription?token=jwt
        │
        ▼
Client streams raw audio bytes over WebSocket
        │
        ▼
Client sends "EOS" signal when done speaking
        │
        ▼
SpeechRecognition processes buffered audio
        │
        ▼
{ "transcript": "recognized text" } ──► returned to client
```

---

## 🛡️ Security

| Practice | Implementation |
|---|---|
| **JWT Verification** | All protected routes validate Bearer tokens via `python-jose` |
| **Password Hashing** | `passlib` with `bcryptjs` backend |
| **Secrets in Env** | All API keys and secrets stored in `.env`, never hardcoded |
| **WebSocket Auth** | Token validated before `websocket.accept()` — closes with code 1008 if invalid |
| **Request Timeouts** | HTTPX client enforces a 30s timeout on all Gemini API calls |
| **User Scoping** | All DB queries filter by `user_id` — users can only access their own data |
| **Structured Logging** | Errors logged with context via Python logging, never leaked to client |

> ⚠️ **Before deploying to production:** restrict `allow_origins` in CORS middleware, enforce HTTPS, and rotate all secrets in `.env`.

---

## 🗺️ Core Modules

### `main.py`
App entry point and the largest module. Defines all HTTP routes and the three WebSocket handlers (`/ws/chat`, `/ws/reminders`, `/ws/transcription`). Manages chat persistence, background audio generation tasks, and graceful startup/shutdown events.

### `ai_service.py`
Async wrapper around the Gemini 2.0 Flash REST API via HTTPX. Accepts a message string and an `is_reminder` flag to switch between the chat prompt and the short reminder prompt. Handles timeouts and API error responses cleanly.

### `reminder_scheduler.py`
Pure async scheduler — no cron library required. Sleeps in 30-second chunks until the target IST datetime, then checks the reminder is still pending before generating a Gemini follow-up and broadcasting via WebSocket.

### `websocket_manager.py`
Maintains a global list of active `(user_id, WebSocket)` pairs. `broadcast_reminder()` iterates this list and delivers reminder payloads to the correct user's open connections.

### `text_to_speech.py`
Synchronous generator that streams MP3 chunks from ElevenLabs. Called via `asyncio.to_thread` to avoid blocking the event loop during audio generation.

### `models/` *(via MongoDB collections)*

| Collection | Purpose |
|---|---|
| `users` | Registered user accounts |
| `chats` | Chat sessions with embedded message arrays (last 50 kept) |
| `reminders` | Scheduled reminders with completion state |

---

## 🔮 Future Improvements

- Rate limiting on chat and reminder endpoints (`slowapi`)
- Multi-turn conversation context passed to Gemini for richer replies
- Audio file cleanup job to purge old MP3s from `static/audio/`
- Refresh token support with rotation
- Redis-backed WebSocket manager for horizontal scaling
- Swagger / OpenAPI documentation refinements
- Unit + integration test suite (pytest + httpx)
- Docker + docker-compose for one-command dev setup

---

## 👤 Author

Built with ❤️ by **[Akshat](https://github.com/EzAkshat)**

<a href="https://github.com/EzAkshat">
  <img src="https://img.shields.io/badge/GitHub-EzAkshat-000?style=flat-square&logo=github&logoColor=white&labelColor=111" />
</a>
&nbsp;
<a href="https://www.linkedin.com/in/naik-akshat">
  <img src="https://img.shields.io/badge/LinkedIn-naik--akshat-0077B5?style=flat-square&logo=linkedin&logoColor=white&labelColor=111" />
</a>

---

<div align="center">
  <sub>⭐ Star this repo if it helped you — it means a lot!</sub>
</div>