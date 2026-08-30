/**
 * inference_bridge.cpp
 *
 * JNI Bridge — Production Native Inference Layer
 *
 * PURPOSE:
 *   This file is the C++ layer that sits between the Android Kotlin app
 *   (AAOS Car App Library) and the native inference libraries (llama.cpp + FAISS).
 *
 *   In the PROTOTYPE (this repo), the AAOS app connects via WebSocket to a
 *   Python FastAPI server. In a PRODUCTION vehicle deployment, this file
 *   replaces that network hop entirely — the Kotlin app calls these JNI
 *   functions directly, and the inference happens in the same process.
 *
 * PRODUCTION ARCHITECTURE:
 *
 *   Kotlin app
 *       │
 *       │ JNI (Java Native Interface)
 *       ▼
 *   inference_bridge.cpp  ← YOU ARE HERE
 *       │                 │
 *       │ llama.cpp API   │ FAISS C++ API
 *       ▼                 ▼
 *   libllama.so       libfaiss.so
 *       │                 │
 *       ▼                 ▼
 *   GGUF model        FAISS index
 *   (on /sdcard)      (on /sdcard)
 *
 * BUILD REQUIREMENTS:
 *   - Android NDK r26+
 *   - llama.cpp compiled as a shared library for arm64-v8a
 *   - FAISS compiled for arm64-v8a (CPU-only, no GPU on AAOS targets)
 *   - CMakeLists.txt referencing this file (see: app/src/main/cpp/CMakeLists.txt)
 *
 * USAGE FROM KOTLIN:
 *   // Load the native library
 *   System.loadLibrary("autobrain_native")
 *
 *   // Declare external functions
 *   external fun nativeInit(modelPath: String, indexPath: String): Boolean
 *   external fun nativeQuery(question: String): String
 *   external fun nativeDestroy()
 */

#include <jni.h>
#include <string>
#include <vector>
#include <memory>
#include <android/log.h>

// Production includes (uncomment when llama.cpp + FAISS are linked):
// #include "llama.h"
// #include "faiss/IndexFlat.h"
// #include "faiss/index_io.h"

#define LOG_TAG "AutoBrainNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ---------------------------------------------------------------------------
// Global state — initialized once per app lifecycle
// ---------------------------------------------------------------------------

// In production these would be:
//   static llama_model* g_model = nullptr;
//   static llama_context* g_ctx = nullptr;
//   static faiss::IndexFlatL2* g_index = nullptr;
//   static std::vector<std::string> g_chunks;

static bool g_initialized = false;
static std::string g_model_path;
static std::string g_index_path;

// ---------------------------------------------------------------------------
// Helper: Convert jstring to std::string
// ---------------------------------------------------------------------------
static std::string jstring_to_string(JNIEnv* env, jstring jstr) {
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return result;
}

// ---------------------------------------------------------------------------
// JNI: Initialize the inference engine
//
// Called once when the AAOS app starts (in CarAppService.onCreate or similar).
// Loads model weights and FAISS index into memory.
//
// @param modelPath  Absolute path to the .gguf model file on /sdcard
// @param indexPath  Absolute path to the FAISS .index file on /sdcard
// @return           JNI_TRUE on success, JNI_FALSE on failure
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jboolean JNICALL
Java_com_autobrain_app_InferenceEngine_nativeInit(
        JNIEnv* env,
        jobject /* this */,
        jstring modelPath,
        jstring indexPath) {

    g_model_path = jstring_to_string(env, modelPath);
    g_index_path = jstring_to_string(env, indexPath);

    LOGI("Initializing AutoBrain native engine");
    LOGI("  Model path: %s", g_model_path.c_str());
    LOGI("  Index path: %s", g_index_path.c_str());

    // --- STUB: Production code would do: ---
    //
    // llama_backend_init();
    // llama_model_params model_params = llama_model_default_params();
    // g_model = llama_load_model_from_file(g_model_path.c_str(), model_params);
    // if (!g_model) { LOGE("Failed to load model"); return JNI_FALSE; }
    //
    // llama_context_params ctx_params = llama_context_default_params();
    // ctx_params.n_ctx = 4096;
    // g_ctx = llama_new_context_with_model(g_model, ctx_params);
    //
    // g_index = dynamic_cast<faiss::IndexFlatL2*>(
    //     faiss::read_index(g_index_path.c_str()));
    // if (!g_index) { LOGE("Failed to load FAISS index"); return JNI_FALSE; }
    //
    // LOGI("Engine ready. Index size: %ld vectors", g_index->ntotal);

    g_initialized = true;
    LOGI("AutoBrain native engine stub initialized (prototype mode)");
    return JNI_TRUE;
}

// ---------------------------------------------------------------------------
// JNI: Run a query through the full RAG + LLM pipeline
//
// This is the hot path called on every user message from the AAOS UI.
// Steps:
//   1. Embed the question using a local embedding model (all-MiniLM equivalent in C++)
//   2. Run FAISS nearest-neighbor search → get top-3 chunk indices
//   3. Build LLM prompt with chunks as context
//   4. Run llama_decode() to generate the answer
//   5. Return JSON string: {"answer": "...", "pages": [1, 2, 3]}
//
// @param question  The user's question as a Java String
// @return          JSON string with answer and source pages
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jstring JNICALL
Java_com_autobrain_app_InferenceEngine_nativeQuery(
        JNIEnv* env,
        jobject /* this */,
        jstring question) {

    std::string q = jstring_to_string(env, question);
    LOGI("nativeQuery called: %s", q.c_str());

    if (!g_initialized) {
        std::string error = R"({"answer": "Engine not initialized.", "pages": []})";
        return env->NewStringUTF(error.c_str());
    }

    // --- STUB: Production code would do: ---
    //
    // Step 1: Embed the question
    // std::vector<float> query_embedding = embed(q);  // local embedding model
    //
    // Step 2: FAISS search
    // faiss::idx_t indices[3];
    // float distances[3];
    // g_index->search(1, query_embedding.data(), 3, distances, indices);
    //
    // Step 3: Build prompt
    // std::string context = "";
    // for (int i = 0; i < 3; i++) {
    //     context += g_chunks[indices[i]] + "\n\n";
    // }
    // std::string prompt = "<|system|>You are AutoBrain...<|user|>" + q
    //                    + "\n\nContext:\n" + context + "<|assistant|>";
    //
    // Step 4: Tokenize + decode
    // auto tokens = llama_tokenize(g_model, prompt, true);
    // llama_decode(g_ctx, llama_batch_get_one(tokens.data(), tokens.size(), 0, 0));
    // std::string answer = sample_tokens_until_stop(g_ctx);
    //
    // Step 5: Return JSON
    // std::string result = "{\"answer\": \"" + answer + "\", \"pages\": [1,2,3]}";
    // return env->NewStringUTF(result.c_str());

    // Stub response for prototype testing
    std::string stub_response =
        "{\"answer\": \"[Native stub] In production this answer comes directly from "
        "llama.cpp + FAISS running in the Android NDK layer, with zero network hops.\","
        "\"pages\": [1, 2, 3]}";

    return env->NewStringUTF(stub_response.c_str());
}

// ---------------------------------------------------------------------------
// JNI: Release all native resources
//
// Called in CarAppService.onDestroy() to free model memory.
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT void JNICALL
Java_com_autobrain_app_InferenceEngine_nativeDestroy(
        JNIEnv* /* env */,
        jobject /* this */) {

    LOGI("Destroying AutoBrain native engine");

    // --- STUB: Production code would do: ---
    //
    // if (g_ctx)   { llama_free(g_ctx);         g_ctx   = nullptr; }
    // if (g_model) { llama_free_model(g_model); g_model = nullptr; }
    // if (g_index) { delete g_index;            g_index = nullptr; }
    // llama_backend_free();

    g_initialized = false;
    LOGI("AutoBrain native engine destroyed");
}
