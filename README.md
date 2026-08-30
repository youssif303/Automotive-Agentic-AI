# AutoBrain Lite 🚗🧠

> A fully local, on-premise AI vehicle manual assistant — built to demonstrate
> Agentic AI, RAG pipelines, and Android Automotive OS integration.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![AAOS](https://img.shields.io/badge/AAOS-Car%20App%20Library%201.4-orange.svg)](https://developer.android.com/cars)
[![llama.cpp](https://img.shields.io/badge/LLM-Phi--3%20mini%20GGUF-purple.svg)](https://github.com/ggerganov/llama.cpp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is this?

**AutoBrain Lite** is a 5-day learning project that demonstrates how to build an intelligent, privacy-first vehicle assistant that runs **100% on-premise** — no cloud, no API keys, no data leaving your machine.

Ask it anything about your car manual ("What does the ESP warning light mean?") and it retrieves the exact relevant pages from the PDF, passes them to a local LLM, and generates a precise answer — all in seconds.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                      │
│                                                                  │
│   ┌─────────────────────┐      ┌──────────────────────────────┐ │
│   │    Web Browser UI   │      │   Android Automotive OS App  │ │
│   │  (HTML/CSS/JS SPA)  │      │   (Car App Library / Kotlin) │ │
│   └──────────┬──────────┘      └──────────────┬───────────────┘ │
└──────────────│──────────────────────────────────│───────────────┘
               │ WebSocket                        │ WebSocket
               │ ws://127.0.0.1:8000/ws           │ ws://10.0.2.2:8000/ws
               ▼                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend (Python)                     │
│              backend/main.py — port 8000                        │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                  AutoBrainAgent                          │  │
│   │                  backend/agent.py                        │  │
│   │                                                          │  │
│   │  1. Intent Classifier (LLM decides: RAG or chitchat?)   │  │
│   │  2. FAISS Retriever   (semantic vector search)          │  │
│   │  3. LLM Generator     (Phi-3-mini GGUF via llama.cpp)  │  │
│   │  4. Follow-up Gen     (suggests 2 related questions)    │  │
│   └──────────────────────────────────────────────────────────┘  │
└──────────┬────────────────────────────────────┬─────────────────┘
           │                                    │
           ▼                                    ▼
┌──────────────────────┐          ┌─────────────────────────────┐
│   Vector Store       │          │   Local LLM                 │
│   FAISS Index        │          │   Phi-3-mini-4k-instruct    │
│   all-MiniLM-L6-v2   │          │   Q4 GGUF (2.2 GB)         │
│   (sentence-xformers)│          │   llama-cpp-python          │
│   data/*.faiss       │          │   models/*.gguf             │
└──────────────────────┘          └─────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM Runtime** | llama-cpp-python (llama.cpp) | CPU inference for GGUF models |
| **LLM Model** | Phi-3-mini-4k-instruct Q4 | Small, fast, accurate local model |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Semantic chunk embedding |
| **Vector DB** | FAISS (flat L2) | Sub-millisecond similarity search |
| **PDF Parsing** | PyMuPDF (fitz) | Robust text extraction from manuals |
| **Web Backend** | FastAPI + Uvicorn | WebSocket server + REST API |
| **Web Frontend** | Vanilla HTML/CSS/JS | Dark glassmorphic SPA, no frameworks |
| **AAOS App** | Kotlin + Car App Library | Distraction-safe in-car UI |
| **AAOS Comms** | OkHttp WebSocket | Connects car app to Python backend |

---

## Project Structure

```
autobrain-lite/
├── backend/
│   ├── main.py          <- FastAPI server (WebSocket + upload API)
│   ├── agent.py         <- AutoBrainAgent: classify -> retrieve -> answer
│   ├── query.py         <- FAISS Retriever class
│   ├── ingest.py        <- PDF parser, chunker, embedder, index builder
│   ├── chat.py          <- CLI chatbot for testing
│   └── requirements.txt
│
├── frontend/
│   ├── index.html       <- Chat UI shell
│   ├── style.css        <- Dark glassmorphic theme
│   └── app.js           <- WebSocket client + upload handler
│
├── aaos-app/
│   ├── settings.gradle.kts
│   ├── build.gradle.kts
│   └── app/src/main/
│       ├── AndroidManifest.xml
│       └── java/com/autobrain/app/
│           ├── AutoBrainCarService.kt  <- AAOS entry point
│           ├── MainCarSession.kt       <- Session lifecycle
│           ├── ChatScreen.kt           <- Main chat UI (ListTemplate)
│           └── InputScreen.kt          <- Voice/keyboard input
│
├── native/
│   └── inference_bridge.cpp  <- JNI bridge stub (production path)
│
├── data/
│   └── <manual-name>/        <- Auto-generated FAISS index per manual
│
├── models/
│   └── *.gguf               <- Downloaded LLM model files
│
└── docs/
    └── architecture.md      <- C4 architecture diagrams
```

---

## Quickstart

### Prerequisites
- Python 3.10+
- A C++ compiler (MinGW on Windows, GCC on Linux/Mac) for llama-cpp-python
- A vehicle manual PDF

### 1. Clone and Set Up Environment
```bash
git clone https://github.com/YOUR_USERNAME/autobrain-lite.git
cd autobrain-lite
python -m venv venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # Linux/Mac
pip install -r backend/requirements.txt
```

### 2. Download the LLM Model
```bash
python backend/download_model.py
# Downloads Phi-3-mini-4k-instruct-q4.gguf (~2.2 GB) into models/
```

### 3. Ingest Your Vehicle Manual
```bash
python backend/ingest.py --pdf your-manual.pdf --output data/your-manual
```

### 4. Start the Server
```bash
python backend/main.py
# Server available at http://127.0.0.1:8000
```

### 5. Open the Web UI
Navigate to **http://127.0.0.1:8000** in your browser.

---

## Android Automotive Setup

1. Install Android Studio with the **Automotive system image** via AVD Manager
2. Open `aaos-app/` as a project in Android Studio
3. Run on the **Automotive emulator** (API 29+)
4. The app connects to `ws://10.0.2.2:8000/ws` (emulator host alias)

---

## Key Concepts Learned

### RAG (Retrieval-Augmented Generation)
Instead of relying on the LLM's training data, we:
1. **Chunk** the PDF into 500-character overlapping segments
2. **Embed** each chunk into a 384-dimensional vector
3. At query time, **embed the question** and find nearest neighbors in FAISS
4. **Inject** the top-3 chunks into the LLM prompt as grounding context

### Agentic Pipeline
The `AutoBrainAgent` first classifies intent before deciding whether to use RAG:
1. Asks the LLM: "Does this require looking up the manual? YES/NO"
2. If YES: runs FAISS retrieval, then generates an answer
3. Always generates 2 follow-up questions for exploration

### AAOS Car App Library
AAOS apps cannot use free-form layouts. Enforced constraints:
- Pre-approved templates only (ListTemplate, SearchTemplate, etc.)
- Max 6 visible list items while driving
- Keyboard disabled when vehicle is moving
- Simplified navigation to minimize distraction

---

## Skills Demonstrated

- Android Automotive / AAOSP — Car App Library, session lifecycle, safety rules
- Systems Engineering / C++ — JNI bridge stub, llama.cpp architecture
- Agentic AI — Intent classification, multi-step reasoning, follow-up generation
- On-Premise AI / RAG — FAISS vector store, semantic chunking, local inference
- Software Architecture — Layered design, C4 diagrams, separation of concerns

---

## Production Roadmap

| Feature | Status |
|---|---|
| Basic RAG pipeline | Done |
| Local LLM inference | Done |
| Web chatbot UI | Done |
| AAOS app scaffold | Done |
| GPU acceleration (CUDA/Metal) | Next |
| JNI native bridge (Android NDK) | Next |
| Voice input (SpeechRecognizer) | Next |
| OBD-II live sensor data injection | Future |

---

## License

MIT License
