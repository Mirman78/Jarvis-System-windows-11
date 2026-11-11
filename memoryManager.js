const fs = require('fs');
const path = require('path');
const { app } = require('electron');

const USER_DATA_DIR = app.getPath('userData');
const CHAT_DATA_DIR = path.join(USER_DATA_DIR, 'chat_data');
const LEGACY_CHAT_DATA_DIR = path.join(__dirname, 'chat_data');

// One-time migration: prior versions stored sessions next to __dirname,
// which is read-only inside a packaged app's asar archive. Copy any
// existing session files into userData so users don't "lose" their chats
// when they upgrade from a dev build to a packaged one.
function migrateLegacyDataIfNeeded() {
  if (!fs.existsSync(LEGACY_CHAT_DATA_DIR)) return;
  if (path.resolve(LEGACY_CHAT_DATA_DIR) === path.resolve(CHAT_DATA_DIR)) return;
  try {
    const legacyFiles = fs.readdirSync(LEGACY_CHAT_DATA_DIR).filter((f) => f.endsWith('.json'));
    if (legacyFiles.length === 0) return;
    fs.mkdirSync(CHAT_DATA_DIR, { recursive: true });
    let migrated = 0;
    for (const f of legacyFiles) {
      const src = path.join(LEGACY_CHAT_DATA_DIR, f);
      const dst = path.join(CHAT_DATA_DIR, f);
      if (!fs.existsSync(dst)) {
        fs.copyFileSync(src, dst);
        migrated++;
      }
    }
    if (migrated > 0) {
      console.log(`📦 Migrated ${migrated} legacy chat file(s) → ${CHAT_DATA_DIR}`);
    }
  } catch (e) {
    console.warn(`⚠️ Could not migrate legacy chat_data: ${e.message}`);
  }
}

function safeSessionId(sessionId) {
  const raw = String(sessionId || 'default_session');
  const cleaned = raw.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 64);
  return cleaned || 'default_session';
}

class StorageHandler {
  constructor(directory) {
    this.directory = directory;
    fs.mkdirSync(this.directory, { recursive: true });
  }

  _getPath(sessionId) {
    return path.join(this.directory, safeSessionId(sessionId) + '.json');
  }

  saveSession(session) {
    const filePath = this._getPath(session.sessionId);
    const tempPath = filePath + '.tmp';
    fs.writeFileSync(tempPath, JSON.stringify(session, null, 2), 'utf-8');
    fs.renameSync(tempPath, filePath);
    return true;
  }

  loadSession(sessionId) {
    const filePath = this._getPath(sessionId);
    if (!fs.existsSync(filePath)) return null;
    try {
      return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    } catch (err) {
      console.error('Failed to parse session', sessionId, err.message);
      return null;
    }
  }

  getAllSessions() {
    const files = fs.readdirSync(this.directory);
    const sessions = [];
    for (const file of files) {
      if (!file.endsWith('.json')) continue;
      const session = this.loadSession(file.replace(/\.json$/, ''));
      if (session) sessions.push(session);
    }
    return sessions;
  }

  cleanOldSessions(maxAgeDays = 30) {
    const now = Date.now();
    const files = fs.readdirSync(this.directory);
    let deleted = 0;
    for (const file of files) {
      if (!file.endsWith('.json')) continue;
      const filePath = path.join(this.directory, file);
      try {
        const stats = fs.statSync(filePath);
        const age = (now - stats.mtimeMs) / (1000 * 60 * 60 * 24);
        if (age > maxAgeDays) {
          fs.unlinkSync(filePath);
          deleted++;
        }
      } catch (err) {
        console.warn('Could not clean session file:', file, err.message);
      }
    }
    return deleted;
  }
}

class ChatMemoryManager {
  constructor(storageHandler) {
    this.storage = storageHandler;
    this.activeSessions = new Map();
    this.maxHistoryLength = 40;
  }

  getOrCreateSession(sessionId, title) {
    const id = safeSessionId(sessionId);
    if (this.activeSessions.has(id)) {
      return this.activeSessions.get(id);
    }

    let session = this.storage.loadSession(id);
    if (!session) {
      session = {
        sessionId: id,
        title: title || 'New Chat',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        messages: []
      };
    }
    if (!Array.isArray(session.messages)) session.messages = [];
    this.activeSessions.set(id, session);
    return session;
  }

  addMessage(sessionId, role, content) {
    const session = this.getOrCreateSession(sessionId);
    const text = String(content == null ? '' : content).slice(0, 80_000);
    session.messages.push({
      id: Date.now().toString(36) + Math.random().toString(36).slice(2, 7),
      role: role === 'user' || role === 'assistant' || role === 'system' ? role : 'assistant',
      content: text,
      timestamp: new Date().toISOString()
    });
    session.updatedAt = new Date().toISOString();
    if (session.messages.length > this.maxHistoryLength) {
      session.messages = session.messages.slice(-this.maxHistoryLength);
    }
    this.storage.saveSession(session);
    return session;
  }

  getHistory(sessionId) {
    return this.getOrCreateSession(sessionId).messages;
  }

  getChatMessages(sessionId, limit = 16) {
    return this.getHistory(sessionId)
      .filter((msg) => msg.role === 'user' || msg.role === 'assistant')
      .slice(-limit)
      .map((msg) => ({ role: msg.role, content: msg.content }));
  }

  deleteSession(sessionId) {
    const filePath = this.storage._getPath(sessionId);
    if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
    this.activeSessions.delete(safeSessionId(sessionId));
    return true;
  }
}

migrateLegacyDataIfNeeded();
const storageHandler = new StorageHandler(CHAT_DATA_DIR);
module.exports = new ChatMemoryManager(storageHandler);