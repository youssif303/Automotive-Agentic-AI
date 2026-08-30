package com.autobrain.app

import androidx.car.app.CarContext
import androidx.car.app.Screen
import androidx.car.app.model.*
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.*
import okhttp3.*
import org.json.JSONObject

/**
 * ChatScreen
 *
 * The primary AAOS UI screen. Uses Car App Library's [LongMessageTemplate]
 * and [ListTemplate] to present a distraction-optimized chat interface.
 *
 * Architecture:
 *   User input (keyboard/voice) → OkHttp WebSocket → FastAPI /ws endpoint
 *   → AutoBrainAgent (Python) → FAISS retrieval + Phi-3 inference
 *   → JSON response → rendered in AAOS template
 *
 * AAOS Distraction Optimization Rules applied:
 *   - Max 6 list items visible at once (car app library enforces this)
 *   - No free-form canvas drawing
 *   - Input via Action buttons only (text input allowed when parked)
 *   - Responses are kept short (<200 chars) for safe reading while driving
 *
 * Future Extension (Production):
 *   Replace WebSocket with direct JNI call to native C++ InferenceEngine:
 *     val answer = InferenceEngine.query(userQuestion)  // native bridge
 */
class ChatScreen(carContext: CarContext, private val mainSession: MainCarSession) : Screen(carContext) {

    // ---------------------------------------------------------------------------
    // State
    // ---------------------------------------------------------------------------

    /** The full conversation history: list of Pair(role, text) */
    private val messages = mutableListOf<Pair<String, String>>()

    /** Follow-up questions suggested by the agent */
    private val followUps = mutableListOf<String>()

    /** Whether we are waiting for an LLM response */
    private var isThinking = false

    /** Source pages retrieved from the manual */
    private var lastSources = listOf<Int>()

    init {
        // Setup initial greeting
        messages.add(Pair("system", "Connected to AutoBrain. Ask anything about your vehicle."))

        // Attach callbacks to the active session
        mainSession.onConnectionStateChanged = { connected ->
            if (connected) {
                // Connection restored, no action needed
            } else {
                messages.add(Pair("system", "Disconnected from server."))
                invalidate()
            }
        }

        mainSession.onErrorOccurred = { error ->
            isThinking = false
            messages.add(Pair("system", "Connection error: ${error.message}"))
            invalidate()
        }

        mainSession.onMessageReceived = { text ->
            isThinking = false
            try {
                val json = JSONObject(text)
                val answer = json.optString("answer", "No answer received.")
                
                // Trim answer for safe driving display
                val trimmedAnswer = if (answer.length > 300) answer.take(300) + "..." else answer
                messages.add(Pair("assistant", trimmedAnswer))

                // Extract page sources
                val retrieved = json.optJSONArray("retrieved")
                lastSources = buildList {
                    if (retrieved != null) {
                        for (i in 0 until retrieved.length()) {
                            val chunk = retrieved.getJSONObject(i)
                            add(chunk.getInt("page"))
                        }
                    }
                }.distinct().sorted()

                // Load follow-up questions
                followUps.clear()
                val followUpsArr = json.optJSONArray("follow_ups")
                if (followUpsArr != null) {
                    for (i in 0 until followUpsArr.length()) {
                        followUps.add(followUpsArr.getString(i))
                    }
                }
            } catch (e: Exception) {
                messages.add(Pair("system", "Error reading response: ${e.message}"))
            }
            invalidate() // trigger re-render
        }
    }



    // ---------------------------------------------------------------------------
    // AAOS Screen Rendering
    // ---------------------------------------------------------------------------

    override fun onGetTemplate(): Template {
        // Build the messages list for the AAOS ListTemplate
        val itemListBuilder = ItemList.Builder()

        // Show last 5 messages (AAOS limits visible items for safety)
        val visibleMessages = messages.takeLast(5)
        visibleMessages.forEach { (role, text) ->
            val prefix = when (role) {
                "user" -> "You: "
                "assistant" -> "AutoBrain: "
                else -> ""
            }
            itemListBuilder.addItem(
                Row.Builder()
                    .setTitle(prefix + text)
                    .build()
            )
        }

        // Show thinking indicator
        if (isThinking) {
            itemListBuilder.addItem(
                Row.Builder()
                    .setTitle("AutoBrain is thinking...")
                    .build()
            )
        }

        // Build action strip with follow-up question chips (max 2 for AAOS safety)
        val actionStripBuilder = ActionStrip.Builder()

        // Add up to 2 follow-up suggestion buttons
        followUps.take(2).forEach { suggestion ->
            actionStripBuilder.addAction(
                Action.Builder()
                    .setTitle(suggestion.take(20)) // AAOS limits label length
                    .setOnClickListener {
                        sendQuery(suggestion)
                    }
                    .build()
            )
        }

        // Add source pages info button if available
        if (lastSources.isNotEmpty()) {
            val sourceLabel = "Sources: p.${lastSources.take(3).joinToString(",")}"
            actionStripBuilder.addAction(
                Action.Builder()
                    .setTitle(sourceLabel)
                    .setOnClickListener { /* Could push a detail screen in the future */ }
                    .build()
            )
        }

        // Input action — opens the AAOS keyboard for text input
        val inputAction = Action.Builder()
            .setTitle("Ask")
            .setOnClickListener {
                // Use AAOS SearchTemplate for voice/text input
                screenManager.push(InputScreen(carContext) { query ->
                    sendQuery(query)
                })
            }
            .build()

        return ListTemplate.Builder()
            .setTitle("AutoBrain — Vehicle Assistant")
            .setHeaderAction(Action.APP_ICON)
            .setSingleList(itemListBuilder.build())
            .setActionStrip(actionStripBuilder.addAction(inputAction).build())
            .build()
    }

    // ---------------------------------------------------------------------------
    // Message Sending
    // ---------------------------------------------------------------------------

    private fun sendQuery(query: String) {
        messages.add(Pair("user", query))
        followUps.clear()
        lastSources = emptyList()
        isThinking = true
        invalidate()

        mainSession.sendMessage(query)
    }
}
