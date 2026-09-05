"""
AutoBrain Lite — Agentic RAG Controller

Coordinates:
    Query → Intent Classification → FAISS Retrieval (if needed) → Prompt Compilation → LLM Response

Uses llama-cpp-python for local CPU execution.
"""

import os
import sys
import json
import urllib.request
from pathlib import Path

# Try importing llama_cpp for local edge/on-premise execution
try:
    from llama_cpp import Llama
    HAS_LOCAL_LLM = True
except ImportError:
    HAS_LOCAL_LLM = False

# Ensure project root is in path for relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.query import Retriever


class AutoBrainAgent:
    """
    Coordinates the on-premise RAG execution:
    - Decides whether RAG search is needed (Intent Classification)
    - Fetches context from FAISS (Retrieval)
    - Directs LLM output with system prompt instructions (Generation)
    - Formulates relative follow-up queries (Agent Action)
    
    Supports:
    - Mode A (On-Premise / Edge): Local quantized GGUF via llama-cpp-python (used locally & on AAOS).
    - Mode B (Cloud Demo): Free Cloud API (Groq/Gemini/OpenAI-compatible) when running on Render 512MB RAM.
    """

    def __init__(
        self,
        model_path: str = "models/Phi-3-mini-4k-instruct-q4.gguf",
        index_dir: str = "data",
        context_window: int = 4096
    ):
        """
        Load the Vector Retriever and initialize the LLM (local or cloud fallback).
        """
        print(f"  [INFO] Initializing AutoBrainAgent...")
        
        # Load retriever
        if not os.path.exists(index_dir):
            os.makedirs(index_dir, exist_ok=True)
            
        # Detect actual index directory (data/some-manual/)
        self.index_dir = index_dir
        if not os.path.exists(os.path.join(index_dir, "index.faiss")):
            for subdir in os.listdir(index_dir):
                candidate = os.path.join(index_dir, subdir)
                if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "index.faiss")):
                    self.index_dir = candidate
                    break
        
        print(f"  [INFO] Connecting retriever to database: {self.index_dir}")
        self.retriever = Retriever(self.index_dir) if os.path.exists(os.path.join(self.index_dir, "index.faiss")) else None

        # Determine LLM execution mode
        self.cloud_api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.mode = "cloud" if (not os.path.exists(model_path) or not HAS_LOCAL_LLM) else "local"

        if self.mode == "local":
            print(f"  [INFO] Loading local LLM model from {model_path}...")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=context_window,
                n_threads=None,
                verbose=False
            )
            print(f"  [OK] Local LLM initialized successfully.")
        else:
            print(f"  [INFO] Running in Cloud Demo Mode (low-memory for Render). API Key configured: {bool(self.cloud_api_key)}")
            self.llm = None
        
        print(f"  [OK] Agent initialized successfully (Mode: {self.mode}).")

    def _call_llm(self, prompt: str, max_tokens: int = 300, temperature: float = 0.2, stop: list = None) -> str:
        """Unified caller for local llama.cpp or multi-provider Cloud API (Groq, Gemini, HuggingFace, OpenAI)."""
        if self.mode == "local" and self.llm is not None:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop or ["<|end|>", "<|system|>", "<|assistant|>", "<|user|>"]
            )
            return response["choices"][0]["text"].strip()

        # Cloud API Fallback
        api_key = (
            os.getenv("GROQ_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("HF_TOKEN")
            or ""
        ).strip()

        if not api_key:
            return "AutoBrain Demo: Please configure GROQ_API_KEY or GEMINI_API_KEY in Render environment variables."

        # Provider 1: Google Gemini (key starts with AIzaSy)
        if api_key.startswith("AIzaSy") or os.getenv("GEMINI_API_KEY"):
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": max_tokens,
                        "temperature": temperature
                    }
                }
                req = urllib.request.Request(
                    url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload).encode("utf-8")
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    return res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8")
                print(f"  [Gemini API Error] {e.code}: {err_body}")
                return f"[Gemini Error {e.code}: {err_body[:100]}]"
            except Exception as e:
                print(f"  [Gemini API Exception] {str(e)}")
                return f"[Gemini Exception: {str(e)}]"

        # Provider 2: Groq (high-speed free tier, key starts with gsk_)
        for model_name in ["groq/compound-mini", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]:
            try:
                url = "https://api.groq.com/openai/v1/chat/completions"
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                req = urllib.request.Request(
                    url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "User-Agent": "Mozilla/5.0 AutoBrain"
                    },
                    data=json.dumps(payload).encode("utf-8")
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    return res_data["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8")
                if "model_not_found" in err_body and model_name != "llama-3.1-8b-instant":
                    continue  # Try next model in fallback list
                print(f"  [Groq API Error] {e.code}: {err_body}")
                return f"[Cloud LLM Error {e.code}: {err_body[:100]}]"
            except Exception as e:
                print(f"  [Cloud LLM Exception] {str(e)}")
                return f"[Cloud LLM Exception: {str(e)}]"

    def _classify_intent(self, query: str) -> bool:
        """
        Agentic Step 1: Intent Classification.
        Decides if the query requires looking up the vehicle owner's manual.
        
        Returns:
            True if vehicle manual retrieval is needed, False otherwise.
        """
        prompt = (
            "<|system|>\n"
            "You are an intent classifier. Determine if the user's input is a question about "
            "a vehicle's operation, buttons, lights, settings, specs, or maintenance that would "
            "require checking the owner's manual.\n"
            "Respond with exactly one word: 'YES' or 'NO'. Do not explain.\n"
            "<|end|>\n"
            f"<|user|>\nQuery: \"{query}\"\n<|end|>\n"
            "<|assistant|>\n"
        )
        
        text = self._call_llm(prompt, max_tokens=10, temperature=0.0, stop=["<|end|>"]).upper()
        # Fallback handling
        needs_rag = "YES" in text
        print(f"  [Agent Classification] Needs RAG? {needs_rag} (LLM responded: '{text}')")
        return needs_rag

    def answer_query(self, query: str, top_k: int = 3) -> dict:
        """
        Processes a user query end-to-end.
        
        Returns:
            dict containing:
              - "answer": LLM generated text
              - "retrieved": List of chunks used (or empty list if direct response)
              - "follow_ups": List of suggested questions
        """
        print(f"\n[Agent API] Processing query: \"{query}\"")
        
        # Step 1: Classify intent
        needs_rag = self._classify_intent(query)
        
        retrieved_chunks = []
        context_str = ""
        
        # Step 2: Retrieve context (if classified as vehicle inquiry)
        if needs_rag and self.retriever is not None:
            print(f"  [Agent Retrieval] Searching FAISS index...")
            retrieved_chunks = self.retriever.search(query, top_k=top_k)
            
            # Format context block
            context_blocks = []
            for chunk in retrieved_chunks:
                context_blocks.append(f"[Page {chunk['page']}]: {chunk['text']}")
            context_str = "\n---\n".join(context_blocks)
            print(f"  [Agent Retrieval] Retrieved {len(retrieved_chunks)} relevant manual chunks.")

        # Step 3: Prompt formulation
        if needs_rag:
            system_prompt = (
                "You are the AutoBrain assistant, an expert on-device assistant for this vehicle.\n"
                "Use the following excerpts from the owner's manual to answer the user's question.\n"
                "RULES:\n"
                "1. Base your answer ONLY on the provided manual excerpts below.\n"
                "2. Cite the page numbers in your response when referencing facts (e.g. '[Page 5]').\n"
                "3. If the answer cannot be found in the excerpts, state clearly: 'I could not find that in the manual.'\n\n"
                f"MANUAL EXCERPTS:\n{context_str}"
            )
        else:
            system_prompt = (
                "You are the AutoBrain assistant, an expert on-device assistant for this vehicle.\n"
                "Answer the user's general greeting or query politely. Keep it brief since "
                "the user is driving."
            )

        prompt = (
            f"<|system|>\n{system_prompt}\n<|end|>\n"
            f"<|user|>\n{query}\n<|end|>\n"
            "<|assistant|>\n"
        )

        # Step 4: Generation
        print(f"  [Agent Generation] Generating response...")
        answer = self._call_llm(
            prompt,
            max_tokens=300,
            temperature=0.2,
            stop=["<|end|>", "<|system|>", "<|assistant|>", "<|user|>"]
        )
        print(f"  [Agent Generation] Response completed.")

        # Step 5: Follow-up questions
        follow_ups = self._generate_follow_ups(query, answer)

        return {
            "answer": answer,
            "retrieved": retrieved_chunks,
            "follow_ups": follow_ups
        }

    def _generate_follow_ups(self, original_query: str, answer: str) -> list:
        """
        Agentic Step 3: Suggests 2 natural follow-up questions.
        """
        prompt = (
            "<|system|>\n"
            "You are an automotive assistant. Based on the user's question and your answer, "
            "suggest exactly 2 brief, relevant follow-up questions the driver might want to ask next.\n"
            "Format: One question per line. No numbers or bullet points. Keep each under 10 words.\n"
            "<|end|>\n"
            f"<|user|>\nQuestion: {original_query}\nAnswer: {answer}\n<|end|>\n"
            "<|assistant|>\n"
        )

        raw = self._call_llm(
            prompt,
            max_tokens=60,
            temperature=0.3,
            stop=["<|end|>", "<|system|>", "<|assistant|>", "<|user|>"]
        )
        
        lines = raw.strip().split("\n")
        follow_ups = [
            line.strip().lstrip("123456789.- ").strip()
            for line in lines
            if line.strip() and not line.strip().startswith("<|")
        ][:2]
        
        return follow_ups
