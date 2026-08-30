package com.autobrain.app

import android.util.Log
import androidx.car.app.CarAppService
import androidx.car.app.Session
import androidx.car.app.validation.HostValidator

/**
 * AutoBrainCarService
 *
 * Entry point for the Car App Library. The AAOS host binds to this service
 * when the user launches the AutoBrain app from the vehicle's infotainment screen.
 *
 * Architecture note:
 *   In a production AAOS system, this service would also initialize the native
 *   C++ JNI bridge (llama.cpp + FAISS) here. For this prototype, it delegates
 *   inference to the local FastAPI WebSocket server running on the dev machine.
 */
class AutoBrainCarService : CarAppService() {

    override fun onCreate(): Unit {
        super.onCreate()
        Log.d("AutoBrain", "CarAppService onCreate")
    }

    /**
     * Returns the host validator. Using ALLOW_ALL_HOSTS for development.
     * In production, restrict to known OEM hosts for security.
     */
    override fun createHostValidator(): HostValidator {
        return HostValidator.ALLOW_ALL_HOSTS_VALIDATOR
    }

    /**
     * Called by the AAOS host when a new session is needed.
     * Each driving session gets its own [MainCarSession] instance.
     */
    override fun onCreateSession(): Session {
        Log.d("AutoBrain", "CarAppService onCreateSession")
        return MainCarSession()
    }
}
