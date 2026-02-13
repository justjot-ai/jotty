/**
 * WhatsApp Web Bridge for Jotty
 * =============================
 *
 * Uses @whiskeysockets/baileys for WhatsApp Web connection.
 * Pure WebSocket — no Puppeteer/Chromium needed.
 * Communicates with Python via stdin/stdout JSON messages.
 *
 * Setup: npm install @whiskeysockets/baileys qrcode-terminal pino
 */

const baileys = require('@whiskeysockets/baileys');
const { default: makeWASocket, useMultiFileAuthState, DisconnectReason, fetchLatestBaileysVersion, Browsers } = baileys;
const qrcode = require('qrcode-terminal');

// Manual in-memory store (Baileys 7.x does not export makeInMemoryStore)
const store = { chats: {}, messages: {} };
const fs = require('fs');
const path = require('path');
const readline = require('readline');
const pino = require('pino');

// Session storage path
const SESSION_PATH = path.join(process.env.HOME || '/tmp', '.jotty', 'whatsapp_session');

// Ensure session directory exists
fs.mkdirSync(SESSION_PATH, { recursive: true });

// Logger (silent for clean stdout JSON)
const logger = pino({ level: 'silent' });

let sock = null;

// Send JSON message to Python
function sendToPython(type, data) {
    const message = JSON.stringify({ type, ...data });
    process.stdout.write(message + '\n');
}

async function startClient() {
    const credsPath = path.join(SESSION_PATH, 'creds.json');
    const hasSavedSession = fs.existsSync(credsPath);
    sendToPython('starting', { status: 'initializing', has_saved_session: hasSavedSession });

    const { state, saveCreds } = await useMultiFileAuthState(SESSION_PATH);
    const { version } = await fetchLatestBaileysVersion();

    sock = makeWASocket({
        version,
        auth: state,
        logger,
        printQRInTerminal: false,
        syncFullHistory: true,
        browser: (Browsers && Browsers.macOS) ? Browsers.macOS('Desktop') : ['Jotty', 'Chrome', '120.0.0']
    });

    // Save credentials on update
    sock.ev.on('creds.update', saveCreds);

    // Connection updates (QR code, connection status)
    sock.ev.on('connection.update', (update) => {
        const { connection, lastDisconnect, qr } = update;

        if (qr) {
            // Display QR in terminal for visual scanning
            qrcode.generate(qr, { small: true }, (qrArt) => {
                process.stderr.write('\n' + qrArt + '\n');
                process.stderr.write('Scan this QR code with WhatsApp on your phone\n\n');
            });
            // Send QR data to Python
            sendToPython('qr', { qr_code: qr });
        }

        if (connection === 'close') {
            const statusCode = lastDisconnect?.error?.output?.statusCode;
            const shouldReconnect = statusCode !== DisconnectReason.loggedOut;

            sendToPython('disconnected', {
                reason: lastDisconnect?.error?.message || 'unknown',
                status_code: statusCode
            });

            if (shouldReconnect) {
                sendToPython('reconnecting', { status: 'reconnecting' });
                startClient();
            }
        } else if (connection === 'open') {
            const user = sock.user;
            sendToPython('ready', {
                status: 'connected',
                info: {
                    id: user?.id,
                    name: user?.name,
                    phone: user?.id?.split(':')[0] || user?.id?.split('@')[0]
                }
            });
        }
    });

    // Incoming messages: forward to Python and store for get_chat_messages
    sock.ev.on('messages.upsert', async ({ messages, type: updateType }) => {
        for (const msg of messages) {
            const jid = msg.key.remoteJid;
            if (!jid) continue;
            const isGroup = jid.endsWith('@g.us');
            const body = msg.message?.conversation
                || msg.message?.extendedTextMessage?.text
                || msg.message?.imageMessage?.caption
                || msg.message?.videoMessage?.caption
                || '';
            const msgType = msg.message ? Object.keys(msg.message)[0] : '';

            if (!store.messages[jid]) store.messages[jid] = [];
            store.messages[jid].push({
                key: msg.key,
                message: msg.message,
                messageTimestamp: msg.messageTimestamp
            });
            if (!store.chats[jid]) store.chats[jid] = { id: jid, name: msg.pushName || jid, is_group: isGroup };
            else if (msg.pushName) store.chats[jid].name = msg.pushName;

            if (updateType === 'notify' && !msg.key.fromMe && msg.message) {
                sendToPython('message', {
                    id: msg.key.id,
                    from: jid,
                    to: sock.user?.id,
                    body,
                    type: msgType,
                    timestamp: msg.messageTimestamp,
                    is_group: isGroup,
                    chat_name: msg.pushName || jid,
                    sender_name: msg.pushName || jid,
                    has_media: ['imageMessage', 'videoMessage', 'audioMessage', 'documentMessage'].includes(msgType)
                });
            }
        }
    });

    // Full history sync: WhatsApp pushes past chats/messages when syncFullHistory is true
    sock.ev.on('messaging-history.set', (eventData) => {
        if (eventData.chats) {
            for (const chat of eventData.chats) {
                const id = chat.id || chat.jid || '';
                if (id) {
                    store.chats[id] = {
                        id,
                        name: chat.name || chat.conversationTimestamp ? 'Chat' : id,
                        is_group: id.endsWith('@g.us')
                    };
                }
            }
        }
        if (eventData.messages) {
            for (const msg of eventData.messages) {
                const jid = msg.key?.remoteJid;
                if (!jid) continue;
                if (!store.messages[jid]) store.messages[jid] = [];
                store.messages[jid].push({
                    key: msg.key,
                    message: msg.message,
                    messageTimestamp: msg.messageTimestamp
                });
                if (!store.chats[jid]) store.chats[jid] = { id: jid, name: jid, is_group: jid.endsWith('@g.us') };
            }
        }
    });
}

// Format phone number to JID
function toJid(phone) {
    if (phone.includes('@')) return phone;
    return phone.replace(/[^0-9]/g, '') + '@s.whatsapp.net';
}

// Read commands from Python via stdin
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false
});

rl.on('line', async (line) => {
    try {
        const cmd = JSON.parse(line);

        switch (cmd.action) {
            case 'send_message':
                await handleSendMessage(cmd);
                break;

            case 'send_media':
                await handleSendMedia(cmd);
                break;

            case 'get_chats':
                await handleGetChats(cmd);
                break;

            case 'get_contacts':
                await handleGetContacts(cmd);
                break;

            case 'get_chat_messages':
                await handleGetChatMessages(cmd);
                break;

            case 'fetch_chat_history':
                await handleFetchChatHistory(cmd);
                break;

            case 'store_stats':
                try {
                    let chatCount = 0;
                    if (store && store.chats) {
                        if (typeof store.chats.all === 'function') chatCount = store.chats.all().length;
                        else if (typeof store.chats.size === 'number') chatCount = store.chats.size;
                        else if (typeof store.chats.forEach === 'function') { store.chats.forEach(() => chatCount++); }
                        else chatCount = Object.keys(store.chats).length;
                    }
                    let msgKeys = [];
                    if (store && store.messages) {
                        if (typeof store.messages.keys === 'function') msgKeys = Array.from(store.messages.keys());
                        else if (typeof store.messages.forEach === 'function') store.messages.forEach((_, k) => msgKeys.push(k));
                        else msgKeys = Object.keys(store.messages);
                    }
                    sendToPython('store_stats', { request_id: cmd.request_id, chat_count: chatCount, message_jids: msgKeys.slice(0, 50) });
                } catch (e) {
                    sendToPython('error', { request_id: cmd.request_id, error: e.message });
                }
                break;

            case 'logout':
                await sock.logout();
                sendToPython('logged_out', { status: 'logged_out' });
                break;

            case 'status':
                sendToPython('status', {
                    connected: sock?.user ? true : false,
                    info: sock?.user
                });
                break;

            default:
                sendToPython('error', { error: `Unknown action: ${cmd.action}` });
        }
    } catch (e) {
        sendToPython('error', { error: e.message });
    }
});

async function handleSendMessage(cmd) {
    try {
        const jid = toJid(cmd.to);
        const result = await sock.sendMessage(jid, { text: cmd.message });

        sendToPython('sent', {
            request_id: cmd.request_id,
            success: true,
            message_id: result.key.id,
            to: jid
        });
    } catch (e) {
        sendToPython('sent', {
            request_id: cmd.request_id,
            success: false,
            error: e.message
        });
    }
}

async function handleSendMedia(cmd) {
    try {
        const jid = toJid(cmd.to);
        let content = {};

        if (cmd.file_path) {
            const buffer = fs.readFileSync(cmd.file_path);
            const ext = path.extname(cmd.file_path).toLowerCase();
            const mimeMap = {
                '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
                '.gif': 'image/gif', '.mp4': 'video/mp4', '.mp3': 'audio/mpeg',
                '.ogg': 'audio/ogg', '.pdf': 'application/pdf', '.doc': 'application/msword'
            };
            const mimetype = mimeMap[ext] || 'application/octet-stream';

            if (mimetype.startsWith('image/')) {
                content = { image: buffer, caption: cmd.caption || '' };
            } else if (mimetype.startsWith('video/')) {
                content = { video: buffer, caption: cmd.caption || '' };
            } else if (mimetype.startsWith('audio/')) {
                content = { audio: buffer, mimetype };
            } else {
                content = { document: buffer, mimetype, fileName: path.basename(cmd.file_path) };
            }
        } else if (cmd.url) {
            content = { image: { url: cmd.url }, caption: cmd.caption || '' };
        }

        const result = await sock.sendMessage(jid, content);

        sendToPython('sent', {
            request_id: cmd.request_id,
            success: true,
            message_id: result.key.id,
            to: jid
        });
    } catch (e) {
        sendToPython('sent', {
            request_id: cmd.request_id,
            success: false,
            error: e.message
        });
    }
}

function getChatsFromStore() {
    if (!store || !store.chats) return [];
    return Object.entries(store.chats).map(([id, c]) => ({
        id,
        name: (c && c.name) || id,
        is_group: (c && c.is_group) || id.endsWith('@g.us')
    }));
}

async function handleGetChats(cmd) {
    try {
        const chats = getChatsFromStore();
        sendToPython('chats', {
            request_id: cmd.request_id,
            chats
        });
    } catch (e) {
        sendToPython('error', {
            request_id: cmd.request_id,
            error: e.message
        });
    }
}

/** Wait for next messaging-history.set (or timeout). Messages are merged by existing handler. */
function waitForHistorySet(sock, timeoutMs) {
    return new Promise((resolve) => {
        const handler = () => {
            sock.ev.off('messaging-history.set', handler);
            resolve(true);
        };
        sock.ev.on('messaging-history.set', handler);
        setTimeout(() => {
            sock.ev.off('messaging-history.set', handler);
            resolve(false);
        }, timeoutMs);
    });
}

function getRawMessagesForChat(chatId) {
    if (!store || !store.messages) return [];
    const key = chatId.includes('@') ? chatId : chatId + '@s.whatsapp.net';
    let messages = store.messages[key];
    if (!messages) {
        const gkey = chatId.includes('@g.us') ? chatId : chatId.replace(/@s.whatsapp.net/, '') + '@g.us';
        messages = store.messages[gkey];
    }
    if (!messages || !Array.isArray(messages)) return [];
    return messages.slice().sort((a, b) => (a.messageTimestamp || 0) - (b.messageTimestamp || 0));
}

async function handleFetchChatHistory(cmd) {
    try {
        const chatId = cmd.chat_id || null;
        const maxMessages = Math.min(parseInt(cmd.max_messages, 10) || 500, 1000);
        if (!chatId) {
            sendToPython('fetch_chat_history_done', {
                request_id: cmd.request_id,
                success: false,
                error: 'chat_id required'
            });
            return;
        }
        const jid = chatId.includes('@') ? chatId : chatId + '@g.us';
        let raw = getRawMessagesForChat(jid);
        if (raw.length === 0) {
            sendToPython('fetch_chat_history_done', {
                request_id: cmd.request_id,
                success: false,
                error: 'No messages in store for this chat. Send a message in the group or wait for sync, then try again.'
            });
            return;
        }
        const HISTORY_WAIT_MS = 12000;
        const MAX_ITERATIONS = 30;
        let iterations = 0;
        while (iterations < MAX_ITERATIONS && raw.length < maxMessages) {
            const oldest = raw[0];
            const key = oldest.key || {};
            const ts = oldest.messageTimestamp || 0;
            if (!key.remoteJid || !key.id) {
                sendToPython('fetch_chat_history_done', {
                    request_id: cmd.request_id,
                    success: true,
                    message_count: raw.length,
                    note: 'Stopped: invalid cursor'
                });
                return;
            }
            try {
                await Promise.race([
                    sock.fetchMessageHistory(50, key, ts),
                    new Promise((_, rej) => setTimeout(() => rej(new Error('fetchMessageHistory timeout')), 15000))
                ]);
            } catch (e) {
                sendToPython('fetch_chat_history_done', {
                    request_id: cmd.request_id,
                    success: true,
                    message_count: raw.length,
                    error: e.message
                });
                return;
            }
            const gotEvent = await waitForHistorySet(sock, HISTORY_WAIT_MS);
            const rawNext = getRawMessagesForChat(jid);
            if (rawNext.length <= raw.length && gotEvent) {
                await new Promise(r => setTimeout(r, 1000));
                const rawAgain = getRawMessagesForChat(jid);
                if (rawAgain.length <= raw.length) break;
                raw = rawAgain;
            } else {
                raw = rawNext;
            }
            if (raw.length >= maxMessages) break;
            iterations++;
        }
        sendToPython('fetch_chat_history_done', {
            request_id: cmd.request_id,
            success: true,
            message_count: raw.length
        });
    } catch (e) {
        sendToPython('fetch_chat_history_done', {
            request_id: cmd.request_id,
            success: false,
            error: e.message
        });
    }
}

function getMessagesFromStore(chatId, limit) {
    if (!store || !store.messages) return [];
    const key = chatId.includes('@') ? chatId : chatId + '@s.whatsapp.net';
    let messages = store.messages[key];
    if (!messages) {
        const gkey = chatId.includes('@g.us') ? chatId : chatId.replace(/@s.whatsapp.net/, '') + '@g.us';
        messages = store.messages[gkey];
    }
    if (!messages || !Array.isArray(messages)) return [];
    const sorted = messages.slice().sort((a, b) => (a.messageTimestamp || 0) - (b.messageTimestamp || 0));
    const slice = sorted.slice(-Math.min(limit || 100, sorted.length));
    return slice.map(m => {
        const k = m.key || m;
        const body = m.message?.conversation || m.message?.extendedTextMessage?.text || m.message?.imageMessage?.caption || '';
        return {
            id: k.id,
            from: k.remoteJid || (k.fromMe ? sock?.user?.id : k.participant),
            body,
            timestamp: m.messageTimestamp || 0,
            fromMe: k.fromMe || false,
            chatId: k.remoteJid
        };
    });
}

async function handleGetChatMessages(cmd) {
    try {
        const limit = Math.min(parseInt(cmd.limit, 10) || 100, 500);
        let chatId = cmd.chat_id || null;
        const chatName = (cmd.chat_name || '').toString().trim().toLowerCase();

        if (!chatId && chatName) {
            const chats = getChatsFromStore();
            const match = chats.find(c => (c.name || '').toLowerCase().includes(chatName) || (c.name || '').toLowerCase().replace(/\s+/g, '-') === chatName);
            if (match) chatId = match.id;
        }
        if (!chatId) {
            sendToPython('chat_messages', {
                request_id: cmd.request_id,
                messages: [],
                error: chatName ? `No chat found matching "${cmd.chat_name}"` : 'chat_id or chat_name required'
            });
            return;
        }
        const jid = chatId.includes('@') ? chatId : chatId + '@s.whatsapp.net';
        const messages = getMessagesFromStore(jid, limit);
        sendToPython('chat_messages', {
            request_id: cmd.request_id,
            messages,
            chat_id: jid,
        });
    } catch (e) {
        sendToPython('error', {
            request_id: cmd.request_id,
            error: e.message
        });
    }
}

async function handleGetContacts(cmd) {
    try {
        sendToPython('contacts', {
            request_id: cmd.request_id,
            contacts: [],
            note: 'Contacts are populated as messages arrive with Baileys'
        });
    } catch (e) {
        sendToPython('error', {
            request_id: cmd.request_id,
            error: e.message
        });
    }
}

// Handle process signals
process.on('SIGTERM', () => {
    sendToPython('shutdown', { reason: 'SIGTERM' });
    process.exit(0);
});

process.on('SIGINT', () => {
    sendToPython('shutdown', { reason: 'SIGINT' });
    process.exit(0);
});

// Start
startClient().catch(err => {
    sendToPython('error', { error: err.message });
    process.exit(1);
});
