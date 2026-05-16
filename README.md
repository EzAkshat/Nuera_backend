<div align="center">

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-000?style=flat-square&logo=python&logoColor=4ade80&labelColor=111" />
  <img src="https://img.shields.io/badge/FastAPI-000?style=flat-square&logo=fastapi&logoColor=4ade80&labelColor=111" />
  <img src="https://img.shields.io/badge/MongoDB-000?style=flat-square&logo=mongodb&logoColor=4ade80&labelColor=111" />
  <img src="https://img.shields.io/badge/Gemini_2.0-000?style=flat-square&logo=google&logoColor=4ade80&labelColor=111" />
  <img src="https://img.shields.io/badge/License-ISC-000?style=flat-square&labelColor=111" />
</p>



# 🚀 Nuera Backend

**Real-time AI chat backend with voice capabilities**

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=13&pause=2000&color=6B7280&center=true&vCenter=true&repeat=true&width=500&lines=FastAPI+%2B+WebSocket+%2B+Gemini+2.0+Flash;Voice+I%2FO+%E2%80%94+TTS+%2B+STT;Smart+Reminders+with+AI+follow-up;JWT+Auth+%2B+MongoDB+%2B+Motor" alt="Typing SVG" />


[Overview](#-overview) · [Features](#-core-features) · [Stack](#-tech-stack) · [Quickstart](#-quick-start) · [Structure](#-project-structure) · [API](#-api-overview) · [How It Works](#-how-it-works) · [Security](#️-security)

</div>

---

## 🧭 Overview

A voice-first AI chat server. Users connect over WebSocket, send text or voice, and receive AI-generated responses as both text and synthesized audio. Reminders can be scheduled with natural language and trigger at a specific datetime with a follow-up AI message.

**Built for** — Desktop apps, voice assistants, real-time chat UIs

**Not included** — Frontend client · see [Nuera_App](https://github.com/EzAkshat/Nuera_App)

---

## ✦ Core Features

**💬 Real-time Chat**
- WebSocket-based streaming — text response first, audio URL follows
- Powered by Google Gemini 2.0 Flash with async HTTPX calls
- Last 50 messages per session saved in MongoDB

**🎙️ Voice I/O**
- **Text-to-Speech** — ElevenLabs MP3 chunks served as static files
- **Speech-to-Text** — WAV audio transcribed via Google STT
- Separate `/ws/transcription` endpoint for voice input

**⏰ Smart Reminders**
- Schedule with a datetime + message — AI generates a contextual follow-up at trigger time
- Background async scheduler checks every 30s, broadcasts over `/ws/reminders`
- Full CRUD endpoints for create, list, update, delete

**🔐 Auth & Scoping**
- JWT-based stateless authentication — tokens checked before WebSocket accept
- All data scoped by `user_id` — no cross-user leakage
- Bcrypt password hashing via passlib

---

## 🧰 Tech Stack

| Component | Choice | Why |
|---|---|---|
| **ASGI Server** | Uvicorn | Fast async event loop for WebSocket handling |
| **Framework** | FastAPI | Type-safe routing, auto OpenAPI docs, dependency injection |
| **Database** | MongoDB + Motor | Async driver for non-blocking queries |
| **AI** | Gemini 2.0 Flash | Fast, cheap, reliable for conversational AI |
| **TTS** | ElevenLabs | Natural-sounding voices, streaming API |
| **STT** | SpeechRecognition | Google STT wrapper — requires WAV input |
| **JWT** | python-jose | HS256 signing with bcrypt password hashing |

---

## ⚡ Quick Start

**1 — Clone & enter**

```bash
git clone https://github.com/EzAkshat/Nuera_backend.git
cd Nuera_backend/server
```

**2 — Create virtual environment**

```bash
python -m venv venv

# Mac / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**3 — Install dependencies**

```bash
pip install -r requirements.txt
```

**4 — Create audio directory**

```bash
# Mac / Linux
mkdir -p static/audio

# Windows
mkdir static\audio
```

**5 — Configure** — create `server/.env`

```env
GEMINI_API_KEY=your_gemini_key
ELEVENLABS_API_KEY=your_elevenlabs_key
ELEVENLABS_VOICE_ID=your_elevenlabs_voice_id
JWT_SECRET=your_secret_key
MONGO_URI=mongodb://localhost:27017
```

**6 — Run**

```bash
uvicorn main:app --reload --port 8000
```

```
Server  →  http://localhost:8000
Docs    →  http://localhost:8000/docs
```

---

## 📁 Project Structure

```
server/
├── main.py                     # Routes, WebSocket handlers, lifespan startup
├── ai_service.py               # Gemini 2.0 Flash API wrapper
├── auth.py                     # JWT decode + user dependency
├── config.py                   # Environment vars & system prompts
├── database.py                 # MongoDB client + collection references
├── reminder_scheduler.py       # Async countdown + WebSocket broadcast
├── text_to_speech.py           # ElevenLabs streaming TTS
├── websocket_manager.py        # Active connection registry
└── static/audio/               # Generated MP3 files
```

---

## 🔌 API Overview

### REST Endpoints

| Method | Route | Auth | Purpose |
|---|---|:---:|---|
| `POST` | `/login` | — | Exchange credentials for JWT |
| `GET` | `/user` | ✅ | Fetch current user details |
| `GET` | `/chat` | ✅ | List all chat sessions |
| `DELETE` | `/chat/{chat_id}` | ✅ | Delete a chat session |
| `POST` | `/reminders` | ✅ | Create & schedule a reminder |
| `GET` | `/reminders` | ✅ | List all reminders |
| `PUT` | `/reminders/{id}` | ✅ | Mark reminder as complete |
| `DELETE` | `/reminders/{id}` | ✅ | Delete a reminder |
| `GET` | `/audio/{message_id}` | ✅ | Get TTS audio URL |

### WebSocket Endpoints

| Route | Purpose |
|---|---|
| `/ws/chat?token=<jwt>` | Send text → receive AI response + audio URL |
| `/ws/reminders?token=<jwt>` | Listen for scheduled reminder broadcasts |
| `/ws/transcription?token=<jwt>` | Stream WAV audio → receive transcript |

---

## 🔄 How It Works

### 💬 Chat Flow

```
  Client
    │
    │  connect /ws/chat?token=<jwt>
    ▼
  Server ── validates JWT ──► invalid → drop (code 1008)
    │
    │  { "message": "Hello", "chat_id": null }
    ▼
  create or continue chat session in MongoDB
    │
    ▼
  save user message ──► call Gemini 2.0 Flash (30s timeout)
    │
    ▼
  send ──► { type: "text_response", content, chat_id }
    │
    ▼
  background task ──► ElevenLabs TTS ──► save MP3 to static/audio/
    │
    ▼
  send ──► { type: "audio_ready", audio_url }
    │
    ▼
  Client
```

---

### ⏰ Reminder Scheduling

```
  POST /reminders  { text, date }
    │
    ▼
  save to MongoDB ──► asyncio.create_task()
    │
    ▼
  sleep in 30s chunks ──► poll until target IST datetime
    │
    ▼
  on trigger ──► generate AI follow-up via Gemini
    │
    ├──► mark reminder as completed in DB
    │
    └──► broadcast over /ws/reminders to user
```

---

### 🎙️ Voice Transcription

```
  Client connects to /ws/transcription?token=<jwt>
    │
    ▼
  stream WAV audio bytes  (WAV format with headers required)
    │
    ▼
  client sends text signal ──► "EOS"  (end of speech)
    │
    ▼
  SpeechRecognition.recognize_google()
    │
    ▼
  return ──► { transcript: "recognized text" }
    │
    ▼
  Client
```

---

## 🛡️ Security

| Layer | Implementation |
|---|---|
| **JWT Tokens** | HS256-signed, secret stored in `.env` only |
| **Passwords** | Bcrypt hashed via passlib |
| **WebSocket Auth** | Token validated before `accept()` — dropped with code `1008` if invalid |
| **User Scoping** | Every DB query filters by `user_id` — no cross-user leakage |
| **Request Timeouts** | HTTPX enforces 30s timeout on all Gemini calls |
| **Error Sanitization** | Exceptions logged server-side, generic messages sent to client |

> ⚠️ **Production checklist** — lock down CORS `allow_origins`, enforce HTTPS, add rate limiting via `slowapi`, set up audio file cleanup job, rotate all `.env` secrets.

---

## 🔮 Roadmap

- Rate limiting on chat and reminder endpoints (`slowapi`)
- Multi-turn context window for memory-aware replies
- Audio file TTL cleanup for `static/audio/`
- Redis-backed WebSocket manager for horizontal scaling
- Refresh token rotation
- Pytest integration tests
- Docker + `docker-compose` setup

---

## 👤 Author

**Akshat Naik**

<a href="https://github.com/EzAkshat">
  <img src="https://img.shields.io/badge/GitHub-EzAkshat-000?style=flat-square&logo=github&logoColor=white&labelColor=111" />
</a>
&nbsp;
<a href="https://www.linkedin.com/in/naik-akshat">
  <img src="https://img.shields.io/badge/LinkedIn-naik--akshat-0077B5?style=flat-square&logo=linkedin&logoColor=white&labelColor=111" />
</a>

---

<div align="center">
  <sub>⭐ Star this repo if it saved you time</sub>
</div>