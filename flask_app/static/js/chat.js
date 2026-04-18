const conversationList = document.getElementById('conversation-list');
const newChatBtn = document.getElementById('new-chat-btn');
const deleteChatBtn = document.getElementById('delete-chat-btn');
const chatMessages = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const activeTitle = document.getElementById('active-title');
const activeMeta = document.getElementById('active-meta');
const logoutBtn = document.getElementById('logout-btn');

let activeConversationId = null;
let conversationsCache = [];

function formatTime(value) {
    if (!value) return '';
    const parsed = new Date(value.replace(' ', 'T'));
    if (Number.isNaN(parsed.getTime())) return value;
    return parsed.toLocaleString();
}

function setActiveConversation(conversation) {
    activeConversationId = conversation ? conversation.id : null;
    activeTitle.textContent = conversation ? (conversation.title || 'Untitled') : 'Select a chat';
    activeMeta.textContent = conversation ? `Updated ${formatTime(conversation.updated_at)}` : 'Messages will appear here';
    deleteChatBtn.classList.toggle('hidden', !conversation);

    const buttons = conversationList.querySelectorAll('[data-conversation-id]');
    buttons.forEach(btn => {
        const isActive = String(btn.dataset.conversationId) === String(activeConversationId);
        btn.classList.toggle('bg-slate-100', isActive);
        btn.classList.toggle('border-slate-200', isActive);
    });
}

function renderConversationList(conversations) {
    conversationList.innerHTML = '';
    if (!conversations.length) {
        const empty = document.createElement('div');
        empty.className = 'text-sm text-slate-500 px-2';
        empty.textContent = 'No conversations yet.';
        conversationList.appendChild(empty);
        return;
    }

    conversations.forEach(conv => {
        const button = document.createElement('button');
        button.type = 'button';
        button.dataset.conversationId = conv.id;
        button.className = 'w-full text-left px-3 py-3 rounded-xl border border-transparent hover:border-slate-200 hover:bg-slate-50 transition';

        const title = document.createElement('div');
        title.className = 'text-sm font-medium text-slate-800 truncate';
        title.textContent = conv.title || 'Untitled';

        const meta = document.createElement('div');
        meta.className = 'text-xs text-slate-500 mt-1';
        meta.textContent = formatTime(conv.updated_at || conv.created_at);

        button.appendChild(title);
        button.appendChild(meta);
        button.addEventListener('click', () => {
            setActiveConversation(conv);
            loadMessages(conv.id);
        });
        conversationList.appendChild(button);
    });

    if (activeConversationId) {
        const active = conversations.find(c => String(c.id) === String(activeConversationId));
        if (active) setActiveConversation(active);
    }
}

function addMessageBubble(message) {
    const wrapper = document.createElement('div');
    const isUser = message.sender === 'user';
    wrapper.className = `flex ${isUser ? 'justify-end' : 'justify-start'}`;

    const bubble = document.createElement('div');
    bubble.className = isUser
        ? 'max-w-[80%] bg-slate-800 text-white px-4 py-3 rounded-2xl rounded-tr-sm shadow'
        : 'max-w-[80%] bg-white text-slate-800 px-4 py-3 rounded-2xl rounded-tl-sm shadow';

    const mainText = document.createElement('div');
    mainText.className = 'text-sm whitespace-pre-line';
    mainText.textContent = message.message_text || '';
    bubble.appendChild(mainText);

    if (!isUser && (message.prediction || message.confidence !== null)) {
        const meta = document.createElement('div');
        meta.className = 'text-xs text-slate-400 mt-2';
        const confText = typeof message.confidence === 'number' ? ` | confidence ${(message.confidence * 100).toFixed(1)}%` : '';
        meta.textContent = `${message.prediction || 'Processed'}${confText}`;
        bubble.appendChild(meta);
    }

    wrapper.appendChild(bubble);
    chatMessages.appendChild(wrapper);
}

function renderMessages(messages) {
    chatMessages.innerHTML = '';
    if (!messages.length) {
        const empty = document.createElement('div');
        empty.className = 'text-sm text-slate-500';
        empty.textContent = 'Start the conversation by sending a message.';
        chatMessages.appendChild(empty);
        return;
    }

    messages.forEach(addMessageBubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function loadConversations() {
    try {
        const res = await fetch('/conversations');
        if (res.status === 401) {
            window.location.href = '/login';
            return;
        }
        const data = await res.json();
        conversationsCache = data.conversations || [];
        renderConversationList(conversationsCache);
    } catch (e) {
        conversationList.innerHTML = '<div class="text-sm text-red-500 px-2">Unable to load conversations.</div>';
    }
}

async function loadMessages(conversationId) {
    if (!conversationId) return;
    try {
        const res = await fetch(`/conversations/${conversationId}`);
        if (res.status === 401) {
            window.location.href = '/login';
            return;
        }
        if (!res.ok) {
            renderMessages([]);
            return;
        }
        const data = await res.json();
        renderMessages(data.messages || []);
    } catch (e) {
        renderMessages([]);
    }
}

async function createConversation(messageText) {
    const payload = {};
    if (messageText) payload.message_text = messageText;
    const res = await fetch('/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    if (res.status === 401) {
        window.location.href = '/login';
        return null;
    }
    const data = await res.json();
    if (!res.ok) return null;
    await loadConversations();
    const conversation = data.conversation;
    if (conversation) {
        setActiveConversation(conversation);
    }
    return conversation;
}

async function sendMessage(messageText) {
    if (!messageText || !messageText.trim()) return;

    let conversationId = activeConversationId;
    if (!conversationId) {
        const created = await createConversation(messageText);
        conversationId = created ? created.id : null;
    }
    if (!conversationId) return;

    const res = await fetch(`/conversations/${conversationId}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message_text: messageText })
    });
    if (res.status === 401) {
        window.location.href = '/login';
        return;
    }
    const data = await res.json();
    if (!res.ok) return;

    if (data.user_message) addMessageBubble(data.user_message);
    if (data.system_message) addMessageBubble(data.system_message);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    await loadConversations();
}

async function deleteConversation() {
    if (!activeConversationId) return;
    if (!confirm('Delete this conversation?')) return;

    const res = await fetch(`/conversations/${activeConversationId}`, { method: 'DELETE' });
    if (res.status === 401) {
        window.location.href = '/login';
        return;
    }
    await loadConversations();
    activeConversationId = null;
    setActiveConversation(null);
    renderMessages([]);
}

async function doLogout() {
    try {
        await fetch('/auth/logout', { method: 'POST' });
    } catch (_) {}
    window.location.href = '/login';
}

newChatBtn.addEventListener('click', async () => {
    const conversation = await createConversation();
    if (conversation) {
        renderMessages([]);
    }
});

chatForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const messageText = chatInput.value.trim();
    chatInput.value = '';
    await sendMessage(messageText);
});

deleteChatBtn.addEventListener('click', deleteConversation);
logoutBtn.addEventListener('click', doLogout);

loadConversations();
