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

const { default: makeWASocket, useMultiFileAuthState, DisconnectReason, fetchLatestBaileysVersion } = require('@whiskeysockets/baileys');
const qrcode = require('qrcode-terminal');
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
    sendToPython('starting', { status: 'initializing' });

    const { state, saveCreds } = await useMultiFileAuthState(SESSION_PATH);
    const { version } = await fetchLatestBaileysVersion();

    sock = makeWASocket({
        version,
        auth: state,
        logger,
        printQRInTerminal: false,
        browser: ['Jotty', 'Chrome', '120.0.0']
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

    // Incoming messages
    sock.ev.on('messages.upsert', async ({ messages, type: updateType }) => {
        if (updateType !== 'notify') return;

        for (const msg of messages) {
            if (msg.key.fromMe) continue;
            if (!msg.message) continue;

            const jid = msg.key.remoteJid;
            const isGroup = jid?.endsWith('@g.us') || false;
            const body = msg.message.conversation
                || msg.message.extendedTextMessage?.text
                || msg.message.imageMessage?.caption
                || msg.message.videoMessage?.caption
                || '';
            const msgType = Object.keys(msg.message)[0];

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

async function handleGetChats(cmd) {
    try {
        // Baileys doesn't have a direct getChats - use store or return empty
        sendToPython('chats', {
            request_id: cmd.request_id,
            chats: [],
            note: 'Use message history for chat list with Baileys'
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
