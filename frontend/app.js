/* ==========================================================================
   AutoBrain Lite — WebSocket Client & UI Interactions
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    let ws = null;
    let selectedIndex = "";
    let retrievedExcerpts = []; // Cached retrieved segments for active citations drawer

    // DOM Elements
    const manualList = document.getElementById("manualList");
    const connectionStatus = document.getElementById("connectionStatus");
    const chatMessages = document.getElementById("chatMessages");
    const chatForm = document.getElementById("chatForm");
    const userInput = document.getElementById("userInput");
    const sendBtn = document.getElementById("sendBtn");
    const activeManualTitle = document.getElementById("activeManualTitle");
    const suggestionsContainer = document.getElementById("suggestionsContainer");
    
    // Upload Elements
    const uploadZone = document.getElementById("uploadZone");
    const pdfFileInput = document.getElementById("pdfFileInput");
    const progressContainer = document.getElementById("progressContainer");
    const progressBar = document.getElementById("progressBar");
    const progressStatus = document.getElementById("progressStatus");

    // Inspector Panel Elements
    const inspectorPanel = document.getElementById("inspectorPanel");
    const inspectorBackdrop = document.getElementById("inspectorBackdrop");
    const closeInspectorBtn = document.getElementById("closeInspectorBtn");
    const inspectorContent = document.getElementById("inspectorContent");

    function openInspectorPanel() {
        inspectorPanel.classList.add("open");
        inspectorBackdrop.classList.add("open");
    }

    function closeInspectorPanel() {
        inspectorPanel.classList.remove("open");
        inspectorBackdrop.classList.remove("open");
    }

    // ---------------------------------------------------------------------------
    // Mobile Sidebar Hamburger Menu
    // ---------------------------------------------------------------------------
    const sidebar = document.getElementById("sidebar");
    const sidebarBackdrop = document.getElementById("sidebarBackdrop");
    const hamburgerBtn = document.getElementById("hamburgerBtn");
    const sidebarCloseBtn = document.getElementById("sidebarCloseBtn");

    function openSidebar() {
        sidebar.classList.add("open");
        sidebarBackdrop.classList.add("open");
    }

    function closeSidebar() {
        sidebar.classList.remove("open");
        sidebarBackdrop.classList.remove("open");
    }

    hamburgerBtn.addEventListener("click", openSidebar);
    sidebarCloseBtn.addEventListener("click", closeSidebar);
    sidebarBackdrop.addEventListener("click", closeSidebar);

    // ---------------------------------------------------------------------------
    // Helper: Format Index Folder Names
    // ---------------------------------------------------------------------------
    function formatIndexName(name) {
        return name
            .replace(/[_-]/g, " ")
            .split(" ")
            .map(w => w.charAt(0).toUpperCase() + w.slice(1))
            .join(" ");
    }

    // ---------------------------------------------------------------------------
    // Backend Host Configuration (Local vs Render Cloud)
    // ---------------------------------------------------------------------------
    const isLocalhost = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
    // Check URL param (?backend=https://...) or localStorage or default Render URL
    const urlParams = new URLSearchParams(window.location.search);
    const customBackend = urlParams.get("backend");
    if (customBackend) {
        localStorage.setItem("autobrain_backend", customBackend);
    }
    const RENDER_BACKEND_URL = "https://automotive-agentic-ai.onrender.com";
    const API_BASE = isLocalhost ? "" : (localStorage.getItem("autobrain_backend") || RENDER_BACKEND_URL).replace(/\/$/, "");

    // ---------------------------------------------------------------------------
    // WebSockets Setup
    // ---------------------------------------------------------------------------
    function connectWebSocket() {
        let wsUrl;
        if (isLocalhost) {
            const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
            wsUrl = `${protocol}//${window.location.host}/ws`;
        } else {
            const cleanHost = API_BASE.replace(/^https?:\/\//, "");
            const wsProto = API_BASE.startsWith("https") ? "wss:" : "ws:";
            wsUrl = `${wsProto}//${cleanHost}/ws`;
        }

        updateStatus("connecting");
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log("WebSocket connected to:", wsUrl);
            updateStatus("connected");
            if (selectedIndex) {
                enableChatInput(true);
            }
        };

        ws.onmessage = (event) => {
            removeLoadingBubble();
            const data = JSON.parse(event.data);

            if (data.error) {
                appendMessage("assistant", `System Error: ${data.error}`);
                return;
            }

            // Cache chunks for citations panel
            retrievedExcerpts = data.retrieved || [];
            
            // Render LLM response
            appendMessage("assistant", data.answer, data.retrieved);

            // Render suggested questions
            renderSuggestions(data.follow_ups);
        };

        ws.onclose = () => {
            console.log("WebSocket disconnected");
            updateStatus("disconnected");
            enableChatInput(false);
            // Reconnect after 3 seconds
            setTimeout(connectWebSocket, 3000);
        };

        ws.onerror = (err) => {
            console.error("WebSocket error:", err);
            ws.close();
        };
    }

    function updateStatus(status) {
        connectionStatus.className = "status-badge";
        const dot = connectionStatus.querySelector(".status-dot") || document.createElement("span");
        dot.className = "status-dot";
        const label = connectionStatus.querySelector(".status-label") || document.createElement("span");
        label.className = "status-label";
        
        connectionStatus.appendChild(dot);
        connectionStatus.appendChild(label);

        if (status === "connected") {
            connectionStatus.classList.add("connected");
            label.textContent = "Connected";
        } else if (status === "connecting") {
            connectionStatus.classList.add("disconnected");
            label.textContent = "Connecting...";
        } else {
            connectionStatus.classList.add("disconnected");
            label.textContent = "Offline";
        }
    }

    function enableChatInput(enabled) {
        userInput.disabled = !enabled;
        sendBtn.disabled = !enabled;
        if (enabled) {
            userInput.placeholder = "Ask a question about your car...";
            userInput.focus();
        } else {
            userInput.placeholder = "Awaiting connection...";
        }
    }

    // ---------------------------------------------------------------------------
    // Indices (Manuals) Loader
    // ---------------------------------------------------------------------------
    async function loadIndices() {
        try {
            const res = await fetch(`${API_BASE}/api/indices`);
            const data = await res.json();
            
            manualList.innerHTML = "";
            
            if (data.indices.length === 0) {
                manualList.innerHTML = `<div class="loading-spinner">No manuals ingested yet.</div>`;
                activeManualTitle.textContent = "No manual selected";
                enableChatInput(false);
                return;
            }

            data.indices.forEach(index => {
                const item = document.createElement("div");
                item.className = "manual-item";
                if (index === data.active) {
                    item.classList.add("active");
                    selectedIndex = index;
                    activeManualTitle.textContent = formatIndexName(index);
                    enableChatInput(true);
                }
                item.innerHTML = `
                    <span>${formatIndexName(index)}</span>
                    <span style="font-size:10px; opacity:0.5;">Active</span>
                `;
                
                item.addEventListener("click", () => { selectIndex(index); closeSidebar(); });
                manualList.appendChild(item);
            });
        } catch (err) {
            console.error("Error loading manuals:", err);
            manualList.innerHTML = `<div class="loading-spinner" style="color:#ff3366;">Error loading indices</div>`;
        }
    }

    async function selectIndex(indexName) {
        try {
            const res = await fetch(`${API_BASE}/api/select-index`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ index_name: indexName })
            });
            const data = await res.json();
            if (data.status === "success") {
                loadIndices(); // Refresh list to update active classes
                appendMessage("system", `Active manual switched to: ${formatIndexName(indexName)}`);
            }
        } catch (err) {
            console.error("Error selecting index:", err);
        }
    }

    // ---------------------------------------------------------------------------
    // Chat UI Render Utilities
    // ---------------------------------------------------------------------------
    function appendMessage(role, text, retrieved = []) {
        const messageDiv = document.createElement("div");
        messageDiv.className = `message ${role}`;

        let bubbleHtml = `<div class="message-bubble">${text}</div>`;
        
        // Add citation tags if there are retrieved chunks
        if (role === "assistant" && retrieved && retrieved.length > 0) {
            // Group by unique page numbers
            const uniquePages = [...new Set(retrieved.map(r => r.page))].sort((a,b)=>a-b);
            let tagsHtml = `<div class="citations">`;
            uniquePages.forEach(page => {
                tagsHtml += `<span class="citation-tag" data-page="${page}">Page ${page}</span>`;
            });
            tagsHtml += `</div>`;
            bubbleHtml += tagsHtml;
        }

        messageDiv.innerHTML = bubbleHtml;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Add event listeners to newly created citation tags
        if (role === "assistant" && retrieved && retrieved.length > 0) {
            messageDiv.querySelectorAll(".citation-tag").forEach(tag => {
                tag.addEventListener("click", (e) => {
                    const page = parseInt(e.target.getAttribute("data-page"));
                    openInspector(page);
                });
            });
        }
    }

    function appendLoadingBubble() {
        const loadingDiv = document.createElement("div");
        loadingDiv.className = "message assistant loading-bubble";
        loadingDiv.innerHTML = `
            <div class="message-bubble">
                <div class="typing-dots">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            </div>
        `;
        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function removeLoadingBubble() {
        const bubble = chatMessages.querySelector(".loading-bubble");
        if (bubble) bubble.remove();
    }

    function renderSuggestions(suggestions) {
        suggestionsContainer.innerHTML = "";
        if (!suggestions || suggestions.length === 0) return;

        suggestions.forEach(suggestion => {
            const chip = document.createElement("div");
            chip.className = "suggestion-chip";
            chip.textContent = suggestion;
            chip.addEventListener("click", () => {
                userInput.value = suggestion;
                chatForm.dispatchEvent(new Event("submit"));
            });
            suggestionsContainer.appendChild(chip);
        });
    }

    // ---------------------------------------------------------------------------
    // Citations Drawer Manager
    // ---------------------------------------------------------------------------
    function openInspector(pageNum) {
        inspectorContent.innerHTML = "";
        
        // Find chunks from this page in our cached excerpts
        const pageChunks = retrievedExcerpts.filter(c => c.page === pageNum);
        
        if (pageChunks.length === 0) {
            inspectorContent.textContent = `No reference segments cached for Page ${pageNum}`;
            return;
        }

        pageChunks.forEach((chunk, index) => {
            const card = document.createElement("div");
            card.className = "citation-excerpt-card";
            card.innerHTML = `
                <div class="citation-excerpt-header">Page ${pageNum} (Segment ${index + 1})</div>
                <div>"${chunk.text}"</div>
            `;
            inspectorContent.appendChild(card);
        });

        openInspectorPanel();
    }

    closeInspectorBtn.addEventListener("click", () => {
        closeInspectorPanel();
    });

    inspectorBackdrop.addEventListener("click", () => {
        closeInspectorPanel();
    });

    // ---------------------------------------------------------------------------
    // Input Submissions (WebSocket + HTTP Fallback)
    // ---------------------------------------------------------------------------
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text) return;

        // Clear suggestions chips
        suggestionsContainer.innerHTML = "";

        // Append user query bubble
        appendMessage("user", text);
        userInput.value = "";
        appendLoadingBubble();

        // If WebSocket is connected, use it
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(text);
        } else {
            // HTTP Fallback (for Vercel or when WebSockets are blocked)
            try {
                const res = await fetch(`${API_BASE}/api/chat`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query: text })
                });
                const data = await res.json();
                removeLoadingBubble();
                if (data.answer) {
                    retrievedExcerpts = data.retrieved || [];
                    appendMessage("assistant", data.answer, data.retrieved);
                    renderSuggestions(data.follow_ups);
                } else {
                    appendMessage("assistant", "Sorry, could not process query.");
                }
            } catch (err) {
                removeLoadingBubble();
                appendMessage("assistant", `Connection error: ${err.message}. Please check that the Render backend is active.`);
            }
        }
    });

    // ---------------------------------------------------------------------------
    // PDF Upload Controls
    // ---------------------------------------------------------------------------
    uploadZone.addEventListener("click", () => {
        pdfFileInput.click();
    });

    // File Drag and Drop Support
    uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = "var(--color-accent)";
        uploadZone.style.background = "rgba(0, 210, 255, 0.05)";
    });

    uploadZone.addEventListener("dragleave", () => {
        uploadZone.style.borderColor = "var(--border-glass)";
        uploadZone.style.background = "rgba(255, 255, 255, 0.01)";
    });

    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = "var(--border-glass)";
        uploadZone.style.background = "rgba(255, 255, 255, 0.01)";
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    pdfFileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    async function handleFileUpload(file) {
        if (!file.name.endsWith(".pdf")) {
            alert("Only PDF files are supported");
            return;
        }

        // Show progress bar
        progressContainer.style.display = "block";
        progressBar.style.width = "0%";
        progressStatus.textContent = "Uploading file...";

        const formData = new FormData();
        formData.append("file", file);

        try {
            // Upload file
            const res = await fetch(`${API_BASE}/api/upload`, {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            
            if (res.ok && data.status === "processing") {
                progressBar.style.width = "100%";
                progressStatus.textContent = "Processing manual locally...";
                
                // Poll indices list every 5 seconds until the new index name shows up
                const pollInterval = setInterval(async () => {
                    const checkRes = await fetch(`${API_BASE}/api/indices`);
                    const checkData = await checkRes.json();
                    
                    if (checkData.indices.includes(data.index_name)) {
                        clearInterval(pollInterval);
                        progressContainer.style.display = "none";
                        // Auto-select the newly uploaded index
                        selectIndex(data.index_name);
                    }
                }, 5000);
            } else {
                throw new Error(data.message || "Failed to upload manual");
            }
        } catch (err) {
            console.error("Upload error:", err);
            progressBar.style.width = "0%";
            progressStatus.textContent = "Upload failed.";
            progressStatus.style.color = "#ff3366";
            setTimeout(() => {
                progressContainer.style.display = "none";
                progressStatus.style.color = "var(--color-accent)";
            }, 3000);
        }
    }

    // ---------------------------------------------------------------------------
    // Initial Setup
    // ---------------------------------------------------------------------------
    loadIndices();
    connectWebSocket();
});
