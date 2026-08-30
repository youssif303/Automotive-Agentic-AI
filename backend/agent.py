"""
AutoBrain Lite — Agentic RAG Controller

Coordinates:
    Query → Intent Classification → FAISS Retrieval (if needed) → Prompt Compilation → LLM Response

Uses llama-cpp-python for local CPU execution.
"""

import os
import sys
from pathlib import Path
from llama_cpp import Llama

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
    """

    def __init__(
        self,
        model_path: str = "models/Phi-3-mini-4k-instruct-q4.gguf",
        index_dir: str = "data",
        context_window: int = 4096
    ):
        """
        Load the LLM and the Vector Retriever.
        """
        print(f"  [INFO] Initializing AutoBrainAgent...")
        
        # Load retriever
        if not os.path.exists(index_dir):
            raise FileNotFoundError(f"Index directory '{index_dir}' does not exist.")
            
        # Detect actual index directory (data/some-manual/)
        self.index_dir = index_dir
        if not os.path.exists(os.path.join(index_dir, "index.faiss")):
            for subdir in os.listdir(index_dir):
                candidate = os.path.join(index_dir, subdir)
                if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "index.faiss")):
                    self.index_dir = candidate
                    break
        
        print(f"  [INFO] Connecting retriever to database: {self.index_dir}")
        self.retriever = Retriever(self.index_dir)

        # Load local LLM via llama.cpp
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LLM model not found at '{model_path}'. Run download_model.py first.")
            
        print(f"  [INFO] Loading LLM model into memory (this can take 5-15 seconds)...")
        # n_ctx=context_window sets context limit, n_threads=None auto-detects CPU cores
        self.llm = Llama(
            model_path=model_path,
            n_ctx=context_window,
            n_threads=None,
            verbose=False  # silences heavy C++ internal logging in console
        )
        print(f"  [OK] Agent initialized successfully.")

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
        
        response = self.llm(
            prompt,
            max_tokens=5,
            temperature=0.0,  # greedy decoding for classification stability
            stop=["<|end|>"]
        )
        
        text = response["choices"][0]["text"].strip().upper()
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
        if needs_rag:
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
        response = self.llm(
            prompt,
            max_tokens=300,
            temperature=0.2,  # low temperature for factual RAG responses
            stop=["<|end|>", "<|system|>", "<|assistant|>", "<|user|>"]
        )
        answer = response["choices"][0]["text"].strip()
        print(f"  [Agent Generation] Response completed.")

        # Step 5: Follow-up suggestions (Agent Action)
        follow_ups = []
        if needs_rag and retrieved_chunks:
            follow_ups = self._generate_follow_ups(query, context_str)

        return {
            "answer": answer,
            "retrieved": retrieved_chunks,
            "follow_ups": follow_ups
        }

    def _generate_follow_ups(self, query: str, context: str) -> list[str]:
        """
        Agentic Step 5: Suggested queries based on retrieved context.
        Generates 2 quick questions the driver might want to ask next.
        """
        prompt = (
            "<|system|>\n"
            "Based on the vehicle manual excerpts below, generate exactly 2 short, distinct follow-up "
            "questions a driver might ask next after asking about: \"" + query + "\"\n"
            "Keep questions very short (under 10 words). Print one question per line starting with '-'.\n"
            "Do not output any introduction or extra formatting.\n"
            f"EXCERPTS:\n{context}\n"
            "<|end|>\n"
            "<|user|>\nGenerate follow-up questions.\n<|end|>\n"
            "<|assistant|>\n"
        )
        
        response = self.llm(
            prompt,
            max_tokens=60,
            temperature=0.3,
            stop=["<|end|>", "<|system|>", "<|assistant|>", "<|user|>"]
        )
        
        lines = response["choices"][0]["text"].strip().split("\n")
        questions = []
        for line in lines:
            line = line.strip().lstrip("-* ").strip()
            if line and len(line) > 5 and line.endswith("?"):
                questions.append(line)
                
        # Return at most 2 items
        return questions[:2]
