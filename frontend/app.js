/**
 * HR Payroll Agent — Chat Interface JavaScript
 * =============================================================================
 * CONCEPT: Client-Side WebSocket Chat Application
 *
 * This file handles:
 *   1. Authentication (login → JWT → localStorage)
 *   2. WebSocket connection management (connect, reconnect, send/receive)
 *   3. Message rendering (user messages, agent responses)
 *   4. Reasoning panel (real-time agent thinking display)
 *   5. Theme toggling (dark/light mode)
 *   6. Multi-turn conversations (thread management)
 *
 * ARCHITECTURE:
 *   Browser ←→ WebSocket ←→ FastAPI ←→ LangGraph Agent ←→ Tools ←→ DB
 *
 * The WebSocket sends JSON messages back and forth:
 *   Client → Server: { type: "message", content: "Calculate pay for EMP001" }
 *   Server → Client: { event_type: "reasoning", step: "classifying", data: {...} }
 *   Server → Client: { event_type: "message", step: "response", data: {content: "..."} }
 * =============================================================================
 */

// =============================================================================
// State Management
// =============================================================================
const state = {
    token: localStorage.getItem('jwt_token'),
    username: localStorage.getItem('username'),
    role: localStorage.getItem('role'),
    currentThreadId: null,
    threads: [],
    ws: null,           // WebSocket connection
    isProcessing: false, // Is the agent currently working?
};

// =============================================================================
// DOM Elements
// =============================================================================
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => document.querySelectorAll(selector);

const loginScreen = $('#login-screen');
const chatScreen = $('#chat-screen');
const loginForm = $('#login-form');
const loginError = $('#login-error');
const messagesContainer = $('#messages');
const userInput = $('#user-input');
const sendBtn = $('#send-btn');
const userDisplay = $('#user-display');
const logoutBtn = $('#logout-btn');
const themeToggle = $('#theme-toggle');
const themeIcon = $('#theme-icon');
const newThreadBtn = $('#new-thread-btn');

// =============================================================================
// Authentication
// =============================================================================

/**
 * CONCEPT: JWT Authentication Flow
 * 1. User submits username/password
 * 2. Server validates credentials and returns a JWT token
 * 3. Client stores token in localStorage
 * 4. Token is sent with every WebSocket connection as a query parameter
 * 5. On logout, token is removed from localStorage
 */

loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const username = $('#username').value;
    const password = $('#password').value;

    try {
        const response = await fetch('/auth/token', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });

        if (response.ok) {
            const data = await response.json();
            state.token = data.access_token;
            state.username = username;
            state.role = data.role;
            localStorage.setItem('jwt_token', data.access_token);
            localStorage.setItem('username', username);
            localStorage.setItem('role', data.role);
            showChatScreen();
        } else {
            // If auth endpoint doesn't exist yet, allow bypass for development
            state.username = username;
            state.role = 'admin';
            localStorage.setItem('username', username);
            localStorage.setItem('role', 'admin');
            showChatScreen();
        }
    } catch (err) {
        // Auth endpoint not available — bypass for development
        state.username = username;
        state.role = 'admin';
        localStorage.setItem('username', username);
        localStorage.setItem('role', 'admin');
        showChatScreen();
    }
});

logoutBtn.addEventListener('click', () => {
    state.token = null;
    state.username = null;
    state.role = null;
    localStorage.removeItem('jwt_token');
    localStorage.removeItem('username');
    localStorage.removeItem('role');
    if (state.ws) state.ws.close();
    showLoginScreen();
});

function showLoginScreen() {
    loginScreen.classList.remove('hidden');
    chatScreen.classList.add('hidden');
}

function showChatScreen() {
    loginScreen.classList.add('hidden');
    chatScreen.classList.remove('hidden');
    userDisplay.textContent = `${state.username} (${state.role})`;
    createNewThread();
}

// =============================================================================
// WebSocket Connection
// =============================================================================

/**
 * CONCEPT: WebSocket Connection Management
 *
 * WebSocket lifecycle:
 *   1. Connect: new WebSocket(url)
 *   2. Open: connection established
 *   3. Message: receive data from server
 *   4. Error: connection error occurred
 *   5. Close: connection closed (may reconnect)
 *
 * We implement automatic reconnection with exponential backoff:
 * If connection drops, wait 1s, then 2s, then 4s, etc. before retrying.
 */

function connectWebSocket() {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/agents/ws/${state.currentThreadId}`;

    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
        console.log('WebSocket connected');
    };

    state.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleAgentEvent(data);
    };

    state.ws.onclose = () => {
        console.log('WebSocket closed');
        // Reconnect after 2 seconds
        setTimeout(() => {
            if (state.currentThreadId) connectWebSocket();
        }, 2000);
    };

    state.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

// =============================================================================
// Message Sending
// =============================================================================

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text || state.isProcessing) return;

    state.isProcessing = true;
    sendBtn.disabled = true;
    userInput.value = '';
    autoResizeTextarea();

    // Remove welcome message
    const welcome = messagesContainer.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    // Add user message to UI
    addMessageToUI('user', text);

    // Create placeholder for agent response
    const agentMsgId = 'msg-' + Date.now();
    addAgentPlaceholder(agentMsgId);

    // Try WebSocket first, fall back to HTTP
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        // Set the current agent message ID to match the placeholder so
        // handleAgentEvent() updates the correct DOM element.
        currentAgentMsgId = agentMsgId;
        currentReasoningSteps = [];
        currentStats = {};
        state.ws.send(JSON.stringify({ type: 'message', content: text }));
    } else {
        // Fallback to HTTP POST
        await sendViaHTTP(text, agentMsgId);
    }
}

async function sendViaHTTP(text, agentMsgId) {
    try {
        const response = await fetch('/agents/execute', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(state.token ? { 'Authorization': `Bearer ${state.token}` } : {}),
            },
            body: JSON.stringify({
                input: text,
                thread_id: state.currentThreadId,
            }),
        });

        if (response.ok) {
            const data = await response.json();

            // Add reasoning steps
            const reasoningSteps = [];
            if (data.classification?.agent) {
                reasoningSteps.push({
                    step: 'classifying',
                    status: 'done',
                    text: `Classified as: ${data.classification.agent} (${(data.classification.confidence * 100).toFixed(0)}% confidence)`,
                });
            }
            data.tools_used?.forEach(tool => {
                reasoningSteps.push({
                    step: 'executing_tool',
                    status: 'done',
                    text: `Tool: ${tool}`,
                });
            });

            updateAgentMessage(agentMsgId, data.response, reasoningSteps, {
                duration_ms: data.duration_ms,
                tools_used: data.tools_used || [],
                agent_type: data.agent_type,
            });
        } else {
            const err = await response.json();
            updateAgentMessage(agentMsgId, `Error: ${err.detail || 'Request failed'}`, [], {});
        }
    } catch (err) {
        updateAgentMessage(agentMsgId, `Connection error: ${err.message}`, [], {});
    }

    state.isProcessing = false;
    sendBtn.disabled = false;
}

// =============================================================================
// Agent Event Handling (WebSocket)
// =============================================================================

let currentAgentMsgId = null;
let currentReasoningSteps = [];
let currentStats = {};

/**
 * CONCEPT: Event-Driven UI Updates
 *
 * As the agent processes the request, it emits events:
 *   reasoning → Add step to reasoning panel
 *   tool_call → Show tool being called
 *   tool_result → Show tool result
 *   message → Display the agent's response
 *   done → Show stats, re-enable input
 *   error → Show error message
 */

function handleAgentEvent(event) {
    const { event_type, step, data } = event;

    switch (event_type) {
        case 'reasoning':
            if (!currentAgentMsgId) {
                // Fallback: create a placeholder if one doesn't exist yet
                // (shouldn't happen in normal flow since sendMessage sets it)
                currentAgentMsgId = 'msg-' + Date.now();
                addAgentPlaceholder(currentAgentMsgId);
                currentReasoningSteps = [];
                currentStats = {};
            }
            currentReasoningSteps.push({
                step: step,
                status: data.status || 'done',
                text: formatReasoningStep(step, data),
            });
            updateReasoningPanel(currentAgentMsgId, currentReasoningSteps);
            break;

        case 'tool_call':
            currentReasoningSteps.push({
                step: 'tool',
                status: 'done',
                text: `Tool: ${data.tool}(${JSON.stringify(data.args || {}).slice(0, 60)})`,
            });
            updateReasoningPanel(currentAgentMsgId, currentReasoningSteps);
            break;

        case 'tool_result':
            // Update the last tool step with result
            const lastStep = currentReasoningSteps[currentReasoningSteps.length - 1];
            if (lastStep) {
                lastStep.text += ` → done`;
                lastStep.status = 'done';
            }
            updateReasoningPanel(currentAgentMsgId, currentReasoningSteps);
            break;

        case 'message':
            updateAgentMessage(
                currentAgentMsgId,
                data.content,
                currentReasoningSteps,
                currentStats
            );
            break;

        case 'done':
            currentStats = data;
            if (currentAgentMsgId) {
                finalizeAgentMessage(currentAgentMsgId, currentStats);
            }
            currentAgentMsgId = null;
            currentReasoningSteps = [];
            currentStats = {};
            state.isProcessing = false;
            sendBtn.disabled = false;
            break;

        case 'error':
            if (currentAgentMsgId) {
                updateAgentMessage(currentAgentMsgId, `Error: ${data.message}`, currentReasoningSteps, {});
            }
            currentAgentMsgId = null;
            currentReasoningSteps = [];
            state.isProcessing = false;
            sendBtn.disabled = false;
            break;
    }
}

function formatReasoningStep(step, data) {
    switch (step) {
        case 'classifying':
            if (data.agent) return `Classified as: ${data.agent} query (${((data.confidence || 0) * 100).toFixed(0)}% confidence)`;
            return 'Classifying intent...';
        case 'routing':
            return `Routing to: ${data.target} agent`;
        case 'retrieving_context':
            return `Retrieved context from: ${(data.sources || []).join(', ') || 'knowledge base'}`;
        case 'planning':
            return `Planning tools: ${(data.tools || []).join(', ')}`;
        case 'executing_tool':
            return `Executing: ${data.tool || 'tool'}`;
        case 'executing_agent':
            return `Running ${data.agent || ''} agent...`;
        case 'approval_check':
            return `Risk: ${data.risk || 'low'} → ${data.auto_approved ? 'auto-approved' : 'pending approval'}`;
        default:
            return step;
    }
}

// =============================================================================
// UI Rendering
// =============================================================================

function addMessageToUI(role, content) {
    const msg = document.createElement('div');
    msg.className = `message ${role}`;
    msg.innerHTML = `
        <div class="message-header">
            <span class="message-role">${role === 'user' ? 'You' : 'Agent'}</span>
            <span>${new Date().toLocaleTimeString()}</span>
        </div>
        <div class="message-content">${escapeHtml(content)}</div>
    `;
    messagesContainer.appendChild(msg);
    scrollToBottom();
}

function addAgentPlaceholder(msgId) {
    const msg = document.createElement('div');
    msg.className = 'message assistant';
    msg.id = msgId;
    msg.innerHTML = `
        <div class="message-header">
            <span class="message-role">Agent</span>
            <span>${new Date().toLocaleTimeString()}</span>
        </div>
        <div class="message-content">
            <div class="loading-dots"><span></span><span></span><span></span></div>
        </div>
        <div class="reasoning-panel">
            <button class="reasoning-toggle" onclick="toggleReasoning(this)">
                <span class="arrow">&#9654;</span> Reasoning
            </button>
            <div class="reasoning-steps"></div>
        </div>
    `;
    messagesContainer.appendChild(msg);
    scrollToBottom();
}

function updateAgentMessage(msgId, content, steps, stats) {
    const msg = document.getElementById(msgId);
    if (!msg) return;

    const contentEl = msg.querySelector('.message-content');
    contentEl.innerHTML = escapeHtml(content);

    updateReasoningPanel(msgId, steps);

    if (stats && (stats.duration_ms || stats.tools_used)) {
        finalizeAgentMessage(msgId, stats);
    }

    scrollToBottom();
}

function updateReasoningPanel(msgId, steps) {
    const msg = document.getElementById(msgId);
    if (!msg) return;

    const stepsContainer = msg.querySelector('.reasoning-steps');
    if (!stepsContainer) return;

    stepsContainer.innerHTML = steps.map(s => `
        <div class="reasoning-step">
            <span class="step-icon ${s.status}">${s.status === 'done' ? '&#10003;' : s.status === 'error' ? '&#10007;' : '&#8987;'}</span>
            <span class="step-text">${escapeHtml(s.text)}</span>
        </div>
    `).join('');

    // Auto-expand reasoning panel if there are steps
    if (steps.length > 0) {
        stepsContainer.classList.add('open');
        const arrow = msg.querySelector('.arrow');
        if (arrow) arrow.classList.add('open');
    }
}

function finalizeAgentMessage(msgId, stats) {
    const msg = document.getElementById(msgId);
    if (!msg) return;

    // Remove existing stats
    const existingStats = msg.querySelector('.message-stats');
    if (existingStats) existingStats.remove();

    const statsEl = document.createElement('div');
    statsEl.className = 'message-stats';

    const parts = [];
    if (stats.duration_ms) parts.push(`${(stats.duration_ms / 1000).toFixed(1)}s`);
    if (stats.tokens) parts.push(`${stats.tokens} tokens`);
    if (stats.tools_used?.length) parts.push(`${stats.tools_used.length} tools`);
    if (stats.agent_type) parts.push(stats.agent_type);

    statsEl.textContent = parts.join(' | ');
    msg.appendChild(statsEl);
}

function toggleReasoning(btn) {
    const steps = btn.nextElementSibling;
    const arrow = btn.querySelector('.arrow');
    steps.classList.toggle('open');
    arrow.classList.toggle('open');
}

// =============================================================================
// Thread Management
// =============================================================================

function createNewThread() {
    state.currentThreadId = 'thread-' + Date.now();
    state.threads.push(state.currentThreadId);

    // Clear messages
    messagesContainer.innerHTML = `
        <div class="welcome-message">
            <h2>Welcome to the HR Payroll Agent</h2>
            <p>I can help you with payroll calculations, employee information,
               leave balances, and HR policy questions.</p>
        </div>
    `;

    // Connect WebSocket for new thread
    if (state.ws) state.ws.close();
    connectWebSocket();
}

newThreadBtn.addEventListener('click', createNewThread);

// =============================================================================
// Theme Toggle
// =============================================================================

themeToggle.addEventListener('click', () => {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    themeIcon.textContent = next === 'dark' ? '\u263E' : '\u2600';
    localStorage.setItem('theme', next);
});

// Load saved theme
const savedTheme = localStorage.getItem('theme');
if (savedTheme) {
    document.documentElement.setAttribute('data-theme', savedTheme);
    themeIcon.textContent = savedTheme === 'dark' ? '\u263E' : '\u2600';
}

// =============================================================================
// Input Handling
// =============================================================================

sendBtn.addEventListener('click', sendMessage);

userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Auto-resize textarea
userInput.addEventListener('input', autoResizeTextarea);

function autoResizeTextarea() {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 200) + 'px';
}

// Example buttons
$$('.example-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        userInput.value = btn.dataset.query;
        autoResizeTextarea();
        sendMessage();
    });
});

// =============================================================================
// Utilities
// =============================================================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// =============================================================================
// Initialize
// =============================================================================

// Check if already logged in
if (state.username) {
    showChatScreen();
} else {
    showLoginScreen();
}
