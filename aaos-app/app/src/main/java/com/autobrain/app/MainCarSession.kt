package com.autobrain.app

import androidx.car.app.Screen
import androidx.car.app.Session
import android.content.Intent
import androidx.car.app.CarContext
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.*
import okhttp3.*
import java.util.concurrent.TimeUnit

/**
 * MainCarSession
 *
 * Manages the lifecycle of a single driving session and maintains the persistent
 * WebSocket connection to the local agent, avoiding disconnects during screen navigation.
 */
class MainCarSession : Session() {

    private val wsUrl = "ws://10.0.2.2:8000/ws"

    // Configure client with generous timeouts and active keep-alive pings
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.MINUTES)   // Wait up to 10 minutes for slow CPU inference
        .writeTimeout(10, TimeUnit.MINUTES)
        .pingInterval(20, TimeUnit.SECONDS)  // Send a ping every 20s to prevent idle disconnects
        .build()

    var webSocket: WebSocket? = null
        private set

    val sessionScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    // Callback to notify the active screen when a message arrives
    var onMessageReceived: ((String) -> Unit)? = null
    var onErrorOccurred: ((Throwable) -> Unit)? = null
    var onConnectionStateChanged: ((Boolean) -> Unit)? = null

    private val wsListener = object : WebSocketListener() {
        override fun onOpen(webSocket: WebSocket, response: Response) {
            sessionScope.launch {
                onConnectionStateChanged?.invoke(true)
            }
        }

        override fun onMessage(webSocket: WebSocket, text: String) {
            sessionScope.launch {
                onMessageReceived?.invoke(text)
            }
        }

        override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
            this@MainCarSession.webSocket = null
            sessionScope.launch {
                onErrorOccurred?.invoke(t)
                onConnectionStateChanged?.invoke(false)
            }
        }

        override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
            this@MainCarSession.webSocket = null
            sessionScope.launch {
                onConnectionStateChanged?.invoke(false)
            }
        }
    }

    override fun onCreateScreen(intent: Intent): Screen {
        if (webSocket == null) {
            connectWebSocket()
        }
        return ChatScreen(carContext, this)
    }

    fun connectWebSocket() {
        if (webSocket != null) return
        val request = Request.Builder().url(wsUrl).build()
        webSocket = client.newWebSocket(request, wsListener)
    }

    fun sendMessage(query: String) {
        if (webSocket == null) {
            connectWebSocket()
        }
        webSocket?.send(query)
    }
}
