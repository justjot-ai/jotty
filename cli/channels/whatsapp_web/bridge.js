/**
 * WhatsApp Web Bridge for Jotty
 * =============================
 *
 * Uses whatsapp-web.js to connect to WhatsApp via QR code.
 * Communicates with Python via stdin/stdout JSON messages.
 *
 * Setup: npm install whatsapp-web.js qrcode-terminal
 */

const { Client, LocalAuth, MessageMedia } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const fs = require('fs');
const path = require('path');
const readline = require('readline');

// Session storage path
const SESSION_PATH = path.join(process.env.HOME || '/tmp', '.jotty', 'whatsapp_session');

// Create client with local auth
const client = new Client({
    authStrategy: new LocalAuth({
        dataPath: SESSION_PATH
    }),
    puppeteer: {
        headless: true,
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-accelerated-2d-canvas',
            '--no-first-run',
            '--no-zygote',
            '--disable-gpu'
        ]
    }
});

// Send JSON message to Python
function sendToPython(type, data) {
    const message = JSON.stringify({ type, ...data });
    console.log(message);
}

// QR Code event
client.on('qr', (qr) => {
    // Display QR in terminal
    qrcode.generate(qr, { small: true });

    // Send QR to Python
    sendToPython('qr', { qr_code: qr });
});

// Ready event
client.on('ready', () => {
    sendToPython('ready', {
        status: 'connected',
        info: client.info
    });
});

// Authenticated event
client.on('authenticated', () => {
    sendToPython('authenticated', { status: 'authenticated' });
});

// Auth failure
client.on('auth_failure', (msg) => {
    sendToPython('auth_failure', { error: msg });
});

// Disconnected
client.on('disconnected', (reason) => {
    sendToPython('disconnected', { reason });
});

// Incoming message
client.on('message', async (msg) => {
    // Skip status messages
    if (msg.isStatus) return;

    const contact = await msg.getContact();
    const chat = await msg.getChat();

    sendToPython('message', {
        id: msg.id._serialized,
        from: msg.from,
        to: msg.to,
        body: msg.body,
        type: msg.type,
        timestamp: msg.timestamp,
        is_group: chat.isGroup,
        chat_name: chat.name,
        sender_name: contact.pushname || contact.name || msg.from,
        has_media: msg.hasMedia
    });
});

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
                await client.logout();
                sendToPython('logged_out', { status: 'logged_out' });
                break;

            case 'status':
                sendToPython('status', {
                    connected: client.info ? true : false,
                    info: client.info
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
        // Format phone number
        let chatId = cmd.to;
        if (!chatId.includes('@')) {
            chatId = chatId.replace(/[^0-9]/g, '') + '@c.us';
        }

        const result = await client.sendMessage(chatId, cmd.message);

        sendToPython('sent', {
            request_id: cmd.request_id,
            success: true,
            message_id: result.id._serialized,
            to: chatId
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
        let chatId = cmd.to;
        if (!chatId.includes('@')) {
            chatId = chatId.replace(/[^0-9]/g, '') + '@c.us';
        }

        let media;
        if (cmd.file_path) {
            media = MessageMedia.fromFilePath(cmd.file_path);
        } else if (cmd.url) {
            media = await MessageMedia.fromUrl(cmd.url);
        } else if (cmd.base64) {
            media = new MessageMedia(cmd.mimetype, cmd.base64, cmd.filename);
        }

        const result = await client.sendMessage(chatId, media, {
            caption: cmd.caption || ''
        });

        sendToPython('sent', {
            request_id: cmd.request_id,
            success: true,
            message_id: result.id._serialized,
            to: chatId
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
        const chats = await client.getChats();
        const chatList = chats.slice(0, cmd.limit || 50).map(chat => ({
            id: chat.id._serialized,
            name: chat.name,
            is_group: chat.isGroup,
            unread_count: chat.unreadCount,
            timestamp: chat.timestamp
        }));

        sendToPython('chats', {
            request_id: cmd.request_id,
            chats: chatList
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
        const contacts = await client.getContacts();
        const contactList = contacts
            .filter(c => c.isMyContact)
            .slice(0, cmd.limit || 100)
            .map(c => ({
                id: c.id._serialized,
                name: c.name || c.pushname,
                number: c.number,
                is_business: c.isBusiness
            }));

        sendToPython('contacts', {
            request_id: cmd.request_id,
            contacts: contactList
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

// Start client
sendToPython('starting', { status: 'initializing' });
client.initialize();
