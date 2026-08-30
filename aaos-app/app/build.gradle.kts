plugins {
    id("com.android.application")
}

android {
    namespace = "com.autobrain.app"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.autobrain.app"
        minSdk = 29          // Android 10 — minimum for AAOS
        targetSdk = 34
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    // Car App Library — provides AAOS-safe UI templates
    implementation(libs.androidx.car.app)
    implementation(libs.androidx.car.app.automotive)

    // OkHttp for WebSocket connection to local FastAPI server
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    // Kotlin coroutines for async WebSocket handling
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // Core Android
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
}
