# AutoBrain Lite — Architecture Documentation

## C4 Model: Context Level (Level 1)

```mermaid
C4Context
    title AutoBrain Lite — System Context

    Person(driver, "Driver / User", "Asks questions about their vehicle manual")

    System(autobrain, "AutoBrain Lite", "Local AI assistant that answers vehicle manual questions using RAG and a local LLM")

    System_Ext(pdf, "Vehicle Manual PDF", "Official OEM PDF manual (e.g. Audi A3 2024)")
    System_Ext(hf, "Hugging Face Hub", "One-time model download only (Phi-3-mini GGUF)")

    Rel(driver, autobrain, "Asks questions", "WebSocket / AAOS Car App")
    Rel(autobrain, pdf, "Ingests on first run", "PyMuPDF text extraction")
    Rel(autobrain, hf, "Downloads model once", "HTTPS (huggingface_hub)")
```

---

## C4 Model: Container Level (Level 2)

```mermaid
C4Container
    title AutoBrain Lite — Containers

    Person(driver, "Driver / User")

    Container_Boundary(web, "Web Interface") {
        Container(browser, "Browser SPA", "HTML / CSS / JS", "Glassmorphic dark-theme chat UI. Connects via WebSocket.")
    }

    Container_Boundary(aaos, "AAOS App") {
        Container(carapp, "Car App", "Kotlin / Car App Library", "Distraction-safe ListTemplate + SearchTemplate UI")
    }

    Container_Boundary(backend, "Python Backend") {
        Container(api, "FastAPI Server", "Python / Uvicorn", "WebSocket endpoint, PDF upload API, static file server")
        Container(agent, "AutoBrainAgent", "Python", "Intent classification, FAISS retrieval, LLM generation, follow-up suggestions")
        Container(retriever, "FAISS Retriever", "Python / FAISS", "384-dim vector similarity search over manual chunks")
        Container(llm, "Local LLM", "llama-cpp-python", "Phi-3-mini-4k-instruct Q4 GGUF — runs on CPU")
    }

    ContainerDb(faiss, "Vector Store", "FAISS flat index + JSON chunks", "Per-manual semantic chunk database")
    ContainerDb(models, "Model Store", "GGUF files on disk", "Downloaded LLM weight files")

    Rel(driver, browser, "Uses", "HTTPS")
    Rel(driver, carapp, "Uses", "AAOS touch / voice")
    Rel(browser, api, "Chat messages", "WebSocket ws://")
    Rel(carapp, api, "Chat messages", "WebSocket ws://10.0.2.2:8000")
    Rel(api, agent, "Forwards query")
    Rel(agent, retriever, "Semantic search")
    Rel(agent, llm, "Generate answer")
    Rel(retriever, faiss, "Read vectors")
    Rel(llm, models, "Load weights")
```

---

## C4 Model: Component Level (Level 3) — AutoBrainAgent

```mermaid
C4Component
    title AutoBrainAgent — Internal Components

    Container_Boundary(agent, "AutoBrainAgent (backend/agent.py)") {
        Component(classifier, "Intent Classifier", "Phi-3-mini prompt", "Asks LLM: does this need the manual? Returns YES/NO")
        Component(retriever, "Retriever", "backend/query.py", "Embeds query with all-MiniLM-L6-v2, top-k FAISS search")
        Component(generator, "Answer Generator", "Phi-3-mini prompt", "Injects retrieved chunks as context, generates grounded answer")
        Component(followup, "Follow-up Generator", "Phi-3-mini prompt", "Generates 2 related suggested questions")
    }

    Rel(classifier, retriever, "If YES: trigger retrieval")
    Rel(retriever, generator, "Pass top-3 chunks")
    Rel(generator, followup, "After answer: generate suggestions")
```

---

## Data Flow — Single Query

```mermaid
sequenceDiagram
    participant U as User (Browser / AAOS)
    participant WS as FastAPI WebSocket
    participant A as AutoBrainAgent
    participant F as FAISS Index
    participant L as Phi-3-mini LLM

    U->>WS: "What does the ESP light mean?"
    WS->>A: answer_query(text)

    A->>L: classify_intent(text) → "YES"
    L-->>A: "YES"

    A->>F: embed(text) → similarity search
    F-->>A: top-3 chunks [page 112, 115, 118]

    A->>L: generate(prompt + chunks)
    L-->>A: "The ESP system monitors..."

    A->>L: generate_follow_ups()
    L-->>A: ["How do I disable ESP?", "What triggers ESP?"]

    A-->>WS: {answer, retrieved, follow_ups}
    WS-->>U: JSON response
```

---

## Production JNI Architecture (Future)

In a real AAOS deployment the Python server is replaced by a native C++ library
called directly from the Android NDK, eliminating the HTTP overhead entirely:

```mermaid
flowchart LR
    A["AAOS App\n(Kotlin)"] -->|JNI call| B["inference_bridge.cpp\n(Android NDK)"]
    B -->|C API| C["llama.cpp\n(libllama.so)"]
    B -->|C++ API| D["faiss.so\n(libfaiss.so)"]
    C --> E["GGUF model\non /sdcard"]
    D --> F["FAISS index\non /sdcard"]
```

See [`native/inference_bridge.cpp`](../native/inference_bridge.cpp) for the stub implementation.
