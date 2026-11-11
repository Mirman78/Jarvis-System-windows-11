const { app, BrowserWindow, globalShortcut, ipcMain, BrowserView, dialog } = require('electron');
const http = require('http');
const path = require('path');
const fs = require('fs');
const os = require('os');
const crypto = require('crypto');
const { spawn, execFileSync } = require('child_process');

app.commandLine.appendSwitch('disk-cache-dir', path.join(os.tmpdir(), 'jarvis-hud-cache'));

const memoryManager = require('./memoryManager.js');
const workflowStorage = require('./workflowStorage.js');
const workflowScheduler = require('./workflowScheduler.js');
const toolsStorage = require('./toolsStorage.js');
const toolRunner = require('./toolRunner.js');

process.on('uncaughtException', (error) => {
  console.error('❌ Uncaught Exception:', error);
});
process.on('unhandledRejection', (reason, promise) => {
  console.error('❌ Unhandled Rejection at:', promise, 'reason:', reason);
});

let win;
let videoView = null;
let isVideoOpen = false;
let mapView = null;
let isMapOpen = false;
let skyView = null;
let isSkyOpen = false;

let aiServerProcess = null;
let aiServerReady = false;
let aiServerPort = 8765;
let aiServerHost = '127.0.0.1';
let modelLoaded = false;
let currentModel = null;

let browserView = null;
let isBrowserOpen = false;
let browserHistory = [];
let browserIndex = -1;

let bridgeServer = null;
const bridgePort = 8766;
const bridgeToken = crypto.randomBytes(16).toString('hex');

// Pending confirmation resolvers. Keyed by request id so concurrent
// confirmations (tool creation, file ops, app ops) never cross wires.
const pendingToolConfirmations = new Map();
const pendingFileConfirmations = new Map();
const pendingAppConfirmations = new Map();

function safeSend(channel, ...args) {
  if (win && !win.isDestroyed() && win.webContents && !win.webContents.isDestroyed()) {
    try { win.webContents.send(channel, ...args); } catch (_) { /* window gone */ }
  }
}

function resolvePending(map, payload) {
  let id = null;
  let approved = false;
  if (payload && typeof payload === 'object') {
    id = payload.id || null;
    approved = !!payload.approved;
  } else {
    approved = !!payload;
  }
  let fn = null;
  if (id && map.has(id)) {
    fn = map.get(id);
    map.delete(id);
  } else if (map.size > 0) {
    const k = map.keys().next().value;
    fn = map.get(k);
    map.delete(k);
  }
  if (fn) fn(approved);
}

ipcMain.on('tool-confirmation-response', (e, p) => resolvePending(pendingToolConfirmations, p));
ipcMain.on('file-operation-response',   (e, p) => resolvePending(pendingFileConfirmations,  p));
ipcMain.on('app-operation-response',    (e, p) => resolvePending(pendingAppConfirmations,   p));

let previousCpuInfo = null;
function getCpuUsage() {
  const cpus = os.cpus();
  if (!previousCpuInfo) {
    previousCpuInfo = cpus.map(cpu => cpu.times);
    return 0;
  }
  let totalDiff = 0;
  let idleDiff = 0;
  for (let i = 0; i < cpus.length; i++) {
    const prev = previousCpuInfo[i];
    const curr = cpus[i].times;
    const total = curr.user + curr.nice + curr.sys + curr.idle + curr.irq;
    const totalPrev = prev.user + prev.nice + prev.sys + prev.idle + prev.irq;
    totalDiff += (total - totalPrev);
    idleDiff += (curr.idle - prev.idle);
  }
  previousCpuInfo = cpus.map(cpu => cpu.times);
  if (totalDiff === 0) return 0;
  return Math.round((1 - idleDiff / totalDiff) * 100);
}

function findPython() {
  // Prefer the project venv's python.exe by absolute path. This is the
  // only reliable way to make ai_server.py run against the venv, since
  // the `py -3` launcher ignores VIRTUAL_ENV and always resolves to a
  // system-wide install.
  const venvPython = process.platform === 'win32'
    ? path.join(__dirname, 'venv', 'Scripts', 'python.exe')
    : path.join(__dirname, 'venv', 'bin', 'python');

  if (fs.existsSync(venvPython)) {
    try {
      execFileSync(venvPython, ['-c', 'import sys; print(sys.version)'],
                   { timeout: 5000, windowsHide: true, stdio: 'ignore' });
      return { bin: venvPython, prefix: [] };
    } catch { /* fall through to system candidates */ }
  }

  // Fallback: system Python. `python` first now (respects activated venv
  // via PATH), `py` last since it bypasses venv entirely.
  const candidates = process.platform === 'win32'
    ? [{ bin: 'python', prefix: [] },
       { bin: 'py', prefix: ['-3'] },
       { bin: 'python3', prefix: [] }]
    : [{ bin: 'python3', prefix: [] }, { bin: 'python', prefix: [] }];

  for (const c of candidates) {
    try {
      execFileSync(c.bin, [...c.prefix, '-c', 'import sys; print(sys.version)'],
                   { timeout: 5000, windowsHide: true, stdio: 'ignore' });
      return c;
    } catch { /* try next */ }
  }
  return null;
}

function markServerReady() {
  if (aiServerReady) return;
  aiServerReady = true;
  console.log('✅ AI Server is ready!');
  sendServerStatus();
  safeSend('ai-server-ready');
}

function sendServerStatus() {
  safeSend('server-status-update', {
    ready: aiServerReady,
    running: aiServerProcess !== null,
    modelLoaded,
    currentModel
  });
}

function waitForServerReady(attempts) {
  if (aiServerReady) return;
  if (!aiServerProcess) return;
  if (attempts > 40) {
    console.error('❌ AI Server did not become ready in time');
    safeSend('ai-server-error', 'AI Server did not become ready in time');
    return;
  }
  makeAIServerRequest('/health', 'GET', null, 1500)
    .then((res) => {
      if (res && res.status === 'ok') markServerReady();
      else setTimeout(() => waitForServerReady(attempts + 1), 500);
    })
    .catch(() => {
      setTimeout(() => waitForServerReady(attempts + 1), 500);
    });
}

function startAIServer() {
  if (aiServerProcess) {
    console.log('AI Server already running');
    return;
  }

  const pythonScript = app.isPackaged
    ? path.join(process.resourcesPath, 'ai_server.py')
    : path.join(__dirname, 'ai_server.py');

  if (!fs.existsSync(pythonScript)) {
    console.error('❌ ai_server.py not found!');
    safeSend('ai-server-error', `ai_server.py not found at ${pythonScript}`);
    return;
  }

  const python = findPython();
  if (!python) {
    const msg = 'Could not find a working Python interpreter. Install Python 3 and make sure either "py", "python", or "python3" is on your PATH.';
    console.error('❌ ' + msg);
    safeSend('ai-server-error', msg);
    return;
  }

  console.log(`🚀 Starting AI Server with "${python.bin} ${python.prefix.join(' ')}"...`);

  const args = [
    ...python.prefix,
    '-u',
    pythonScript,
    '--port', aiServerPort.toString(),
    '--host', aiServerHost,
    '--data-dir', app.getPath('userData'),
    '--bridge-port', bridgePort.toString(),
    '--bridge-token', bridgeToken
  ];

  aiServerProcess = spawn(python.bin, args, {
    stdio: ['pipe', 'pipe', 'pipe'],
    env: { ...process.env, PYTHONUNBUFFERED: '1' }
  });

  sendServerStatus();

  aiServerProcess.stdout.on('data', (data) => {
    const output = data.toString().trim();
    console.log(`[AI Server] ${output}`);
    if (output.includes('J.A.R.V.I.S. AI Server started') && !aiServerReady) {
      markServerReady();
    }
  });

  aiServerProcess.stderr.on('data', (data) => {
    const msg = data.toString().trim();
    console.error(`[AI Server Error] ${msg}`);
    safeSend('ai-server-error', msg);
  });

  aiServerProcess.on('close', (code) => {
    console.log(`AI Server process exited with code ${code}`);
    aiServerProcess = null;
    aiServerReady = false;
    modelLoaded = false;
    sendServerStatus();
  });

  aiServerProcess.on('error', (err) => {
    console.error('❌ Failed to start AI Server:', err);
    safeSend('ai-server-error', `Failed to start server: ${err.message}`);
    aiServerProcess = null;
    aiServerReady = false;
    sendServerStatus();
  });

  waitForServerReady(0);
}

function stopAIServer() {
  if (aiServerProcess) {
    console.log('🛑 Stopping AI Server...');
    aiServerProcess.kill();
    aiServerProcess = null;
    aiServerReady = false;
    modelLoaded = false;
    sendServerStatus();
    console.log('✅ AI Server stopped');
  }
}

function createWindow() {
  win = new BrowserWindow({
    width: 1920,
    height: 1080,
    fullscreen: true,
    frame: false,
    backgroundColor: '#0a0a12',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    title: 'J.A.R.V.I.S.'
  });

  win.webContents.setBackgroundThrottling(false);
  win.webContents.session.setPermissionRequestHandler((webContents, permission, callback) => {
    if (permission === 'media') callback(true);
    else callback(false);
  });

  win.loadFile('index.html');

  win.webContents.on('did-finish-load', () => {
    console.log('Window loaded, starting AI Server...');
    setTimeout(() => {
      startAIServer();
    }, 1000);
  });

  // NOTE: Escape is deliberately NOT registered as a *global* shortcut.
  // Global shortcuts intercept the key before the renderer sees it, which
  // meant the confirm-overlay's own Escape handler in index.html could
  // never fire while a view was open. Instead, each view's webContents
  // gets its own `before-input-event` handler below, so Escape only closes
  // a view when that view actually has focus — the HUD keeps working.
  globalShortcut.register('F11', () => {
    win.setFullScreen(!win.isFullScreen());
  });

  globalShortcut.register('CommandOrControl+R', () => {
    win.reload();
  });

  win.on('closed', () => {
    win = null;
    stopAIServer();

    // Resolve any pending confirmation promises as "declined" so nothing hangs.
    for (const fn of pendingToolConfirmations.values()) { try { fn(false); } catch (_) {} }
    for (const fn of pendingFileConfirmations.values()) { try { fn(false); } catch (_) {} }
    for (const fn of pendingAppConfirmations.values())  { try { fn(false); } catch (_) {} }
    pendingToolConfirmations.clear();
    pendingFileConfirmations.clear();
    pendingAppConfirmations.clear();
  });
}

// ==============================================================
// WORKFLOW EXECUTION
// ==============================================================
function executeWorkflow(job) {
  const workflow = job.workflow || {};

  if (workflow.type === 'script') {
    if (!workflow.scriptPath || !fs.existsSync(workflow.scriptPath)) {
      safeSend('workflow-result', {
        id: job.id,
        result: `❌ "${job.name}" failed: script not found at ${workflow.scriptPath}`
      });
      return;
    }
    const python = findPython();
    if (!python) {
      safeSend('workflow-result', {
        id: job.id,
        result: `❌ "${job.name}" failed: no Python interpreter found on PATH.`
      });
      return;
    }
    const proc = spawn(python.bin, [...python.prefix, workflow.scriptPath]);
    let stdout = '';
    let stderr = '';
    proc.stdout.on('data', (d) => { stdout += d.toString(); });
    proc.stderr.on('data', (d) => { stderr += d.toString(); });
    proc.on('close', (code) => {
      const output = stdout.trim() || stderr.trim() || '(no output)';
      const summary = output.length > 2000 ? output.slice(0, 2000) + '... (truncated)' : output;
      safeSend('workflow-result', {
        id: job.id,
        result: code === 0
          ? `▶️ "${job.name}" completed at ${new Date().toLocaleTimeString()}:\n${summary}`
          : `❌ "${job.name}" exited with code ${code}:\n${summary}`
      });
    });
    proc.on('error', (err) => {
      safeSend('workflow-result', {
        id: job.id,
        result: `❌ "${job.name}" failed to start: ${err.message}`
      });
    });
    return;
  }

  if (workflow.type === 'ai-prompt') {
    if (!aiServerProcess || !modelLoaded) {
      safeSend('workflow-result', {
        id: job.id,
        result: `⚠️ "${job.name}" skipped: no model is currently loaded.`
      });
      return;
    }
    makeAIServerRequest('/generate', 'POST', { prompt: workflow.prompt || job.name, max_tokens: 512, use_tools: false }, 0)
      .then((response) => {
        safeSend('workflow-result', {
          id: job.id,
          result: response && response.success
            ? `▶️ "${job.name}" (${new Date().toLocaleTimeString()}):\n${response.response}`
            : `❌ "${job.name}" failed: ${(response && response.error) || 'Unknown error'}`
        });
      })
      .catch((err) => {
        safeSend('workflow-result', {
          id: job.id,
          result: `❌ "${job.name}" failed: ${err.message}`
        });
      });
    return;
  }

  safeSend('workflow-result', {
    id: job.id,
    result: `❌ "${job.name}" has an unrecognized workflow type: ${workflow.type}`
  });
}

app.whenReady().then(() => {
  createWindow();
  startBridgeServer();

  if (typeof toolRunner.initOrchestrator === 'function') {
    toolRunner.initOrchestrator({ port: aiServerPort, token: bridgeToken });
  }

  try {
    const jobs = workflowStorage.loadJobs();
    workflowScheduler.initAll(jobs, executeWorkflow);
    console.log(`⏰ Initialized ${jobs.length} workflow schedule(s).`);
  } catch (err) {
    console.error('Failed to initialize workflow schedules:', err);
  }

  try {
    const cleaned = memoryManager.storage.cleanOldSessions(90);
    if (cleaned > 0) console.log(`🧹 Cleaned ${cleaned} old chat(s) (>90 days).`);
  } catch (e) { console.warn('Session cleanup failed:', e.message); }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  stopAIServer();
  if (process.platform !== 'darwin') app.quit();
});

app.on('will-quit', () => {
  stopAIServer();
  globalShortcut.unregisterAll();
  if (bridgeServer) {
    bridgeServer.close();
    bridgeServer = null;
  }
});

// ==============================================================
// IPC HANDLERS
// ==============================================================

ipcMain.on('minimize', () => { if (win && !win.isDestroyed()) win.minimize(); });
ipcMain.on('close', () => { if (win && !win.isDestroyed()) win.close(); });

ipcMain.handle('get-history', async (event, sessionId) => {
  return memoryManager.getHistory(sessionId || 'default_session');
});

ipcMain.handle('add-message', async (event, data) => {
  const session = memoryManager.addMessage(data.sessionId || 'default_session', data.role, data.content);
  return session.messages;
});

ipcMain.handle('list-sessions', async () => {
  try {
    const sessions = memoryManager.storage.getAllSessions();
    return {
      success: true,
      sessions: sessions
        .map(s => ({
          sessionId: s.sessionId,
          title: s.title || 'New Chat',
          updatedAt: s.updatedAt || s.createdAt || ''
        }))
        .sort((a, b) => String(b.updatedAt).localeCompare(String(a.updatedAt)))
    };
  } catch (err) {
    return { success: false, error: err.message, sessions: [] };
  }
});

ipcMain.handle('create-session', async (event, title) => {
  try {
    const id = 'session_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    const session = memoryManager.getOrCreateSession(id, title || 'New Chat');
    return { success: true, sessionId: session.sessionId, title: session.title };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('delete-session', async (event, id) => {
  try {
    if (!id) return { success: false, error: 'session id required' };
    memoryManager.deleteSession(id);
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('rename-session', async (event, data) => {
  try {
    const id = data && data.id;
    if (!id) return { success: false, error: 'session id required' };
    const title = String((data && data.title) || 'New Chat').slice(0, 80);
    const session = memoryManager.getOrCreateSession(id);
    session.title = title;
    session.updatedAt = new Date().toISOString();
    memoryManager.storage.saveSession(session);
    return { success: true, title };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('ai-server-status', async () => {
  return {
    running: aiServerProcess !== null,
    port: aiServerPort,
    modelLoaded,
    currentModel
  };
});

ipcMain.handle('ai-list-models', async () => {
  if (!aiServerProcess) return { success: false, error: 'AI Server not running' };
  try {
    return await makeAIServerRequest('/models', 'GET');
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('ai-load-model', async (event, data) => {
  if (!aiServerProcess) {
    return { success: false, error: 'AI Server not running' };
  }

  try {
    const cpuCores = os.cpus().length;
    const n_threads = data.n_threads !== undefined ? data.n_threads : Math.min(cpuCores, 12);

    const payload = {
      model: data.model || null,
      model_path: data.modelPath || null,
    };
    for (const k of ['engine','n_gpu_layers','n_ctx','n_threads',
                     'tensor_split','devices','flash_attn','chat_format']) {
      if (data[k] !== undefined) payload[k] = data[k];
    }
    if (payload.n_gpu_layers === undefined) payload.n_gpu_layers = -1;
    if (payload.n_threads === undefined) payload.n_threads = Math.min(os.cpus().length, 12);
    if (payload.n_ctx === undefined) payload.n_ctx = 4096;

    safeSend('ai-model-load-status', { status: 'loading', model: payload.model || payload.model_path });

    const response = await makeAIServerRequest('/load-model', 'POST', payload, 300000);
    if (response.success) {
      modelLoaded = true;
      // FIX: path.basename() throws if given null/undefined. Guard both sides.
      currentModel = payload.model
        || (payload.model_path ? path.basename(payload.model_path) : null);
      safeSend('ai-model-load-status', { status: 'success', model: currentModel });
    } else {
      safeSend('ai-model-load-status', { status: 'failed', error: response.error });
    }
    return response;
  } catch (error) {
    safeSend('ai-model-load-status', { status: 'failed', error: error.message });
    return { success: false, error: error.message };
  }
});

ipcMain.handle('select-model-file', async () => {
  const result = await dialog.showOpenDialog(win, {
    properties: ['openFile'],
    filters: [{ name: 'GGUF Models', extensions: ['gguf'] }]
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  return result.filePaths[0];
});

ipcMain.handle('ai-unload-model', async () => {
  if (!aiServerProcess) {
    return { success: false, error: 'AI Server not running' };
  }
  try {
    const response = await makeAIServerRequest('/unload-model', 'POST');
    if (response.success) {
      modelLoaded = false;
      currentModel = null;
    }
    return response;
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('ai-generate', async (event, data) => {
  if (!aiServerProcess) {
    return { success: false, error: 'AI Server not running' };
  }
  if (!modelLoaded) {
    return { success: false, error: 'No model loaded. Please load a model first.' };
  }
  try {
    return await makeAIServerRequest('/generate', 'POST', data, 0);
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('ai-abort', async () => {
  if (!aiServerProcess) {
    return { success: false, error: 'AI Server not running' };
  }
  try {
    return await makeAIServerRequest('/abort', 'POST', null, 5000);
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('ai-list-memories', async (event, query) => {
  if (!aiServerProcess) return { success: false, error: 'AI Server not running' };
  try {
    const qs = query ? ('?query=' + encodeURIComponent(query)) : '';
    return await makeAIServerRequest('/memories' + qs, 'GET');
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('ai-add-memory', async (event, data) => {
  if (!aiServerProcess) return { success: false, error: 'AI Server not running' };
  try {
    const payload = (typeof data === 'string') ? { fact: data } : (data || {});
    return await makeAIServerRequest('/memory-add', 'POST', payload, 10000);
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('ai-delete-memory', async (event, idOrFact) => {
  if (!aiServerProcess) return { success: false, error: 'AI Server not running' };
  try {
    const payload = typeof idOrFact === 'string' ? { id: idOrFact, fact: idOrFact } : (idOrFact || {});
    return await makeAIServerRequest('/memory-delete', 'POST', payload, 10000);
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('ai-kb-list', async () => {
  if (!aiServerProcess) return { success: false, error: 'AI Server not running' };
  try {
    return await makeAIServerRequest('/kb/documents', 'GET');
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('ai-kb-search', async (event, query) => {
  if (!aiServerProcess) return { success: false, error: 'AI Server not running' };
  try {
    return await makeAIServerRequest('/kb/search?query=' + encodeURIComponent(query || ''), 'GET');
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('ai-kb-add', async (event, filePath) => {
  if (!aiServerProcess) return { success: false, error: 'AI Server not running' };
  try {
    return await makeAIServerRequest('/kb/add', 'POST', { path: filePath }, 0);
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('ai-kb-remove', async (event, match) => {
  if (!aiServerProcess) return { success: false, error: 'AI Server not running' };
  try {
    const payload = typeof match === 'string' ? { id: match, path: match, title: match } : (match || {});
    return await makeAIServerRequest('/kb/remove', 'POST', payload, 10000);
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('ai-browser-navigate', async (event, args) => {
  if (typeof args === 'string') args = { url: args };
  return bridgeNavigate(args || {});
});
ipcMain.handle('ai-browser-click', async (event, args) => bridgeClick(args || {}));
ipcMain.handle('ai-browser-fill', async (event, args) => bridgeFill(args || {}));
ipcMain.handle('ai-browser-read', async (event, args) => bridgeRead(args || {}));
ipcMain.handle('ai-browser-screenshot', async () => bridgeScreenshot());

ipcMain.handle('ai-files-find', async (event, args) => {
  if (!aiServerProcess) return { success: false, error: 'AI Server not running' };
  try {
    return await makeAIServerRequest('/files/find', 'POST', args || {}, 30000);
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('ai-files-list', async (event, directory) => {
  if (!aiServerProcess) return { success: false, error: 'AI Server not running' };
  try {
    return await makeAIServerRequest('/files/list?directory=' + encodeURIComponent(directory || ''), 'GET');
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('ai-files-read', async (event, filePath) => {
  if (!aiServerProcess) return { success: false, error: 'AI Server not running' };
  try {
    return await makeAIServerRequest('/files/read', 'POST', { path: filePath }, 15000);
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('ai-apps-search', async (event, query) => {
  if (!aiServerProcess) return { success: false, error: 'AI Server not running' };
  try {
    return await makeAIServerRequest('/apps/search?query=' + encodeURIComponent(query || ''), 'GET', null, 35000);
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('ai-apps-list', async (event, query) => {
  if (!aiServerProcess) return { success: false, error: 'AI Server not running' };
  try {
    return await makeAIServerRequest('/apps/list?query=' + encodeURIComponent(query || ''), 'GET', null, 35000);
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('ai-web-search', async (event, query, maxResults) => {
  if (!aiServerProcess) return { success: false, error: 'AI Server not running' };
  try {
    const qs = '?query=' + encodeURIComponent(query || '') +
               (maxResults ? '&max_results=' + encodeURIComponent(maxResults) : '');
    return await makeAIServerRequest('/web/search' + qs, 'GET', null, 25000);
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('reconnect-ai-server', async () => {
  stopAIServer();
  setTimeout(() => startAIServer(), 500);
  return { success: true };
});

ipcMain.handle('get-resource-stats', async () => {
  const totalMem = os.totalmem();
  const freeMem = os.freemem();
  const memUsage = Math.round((1 - freeMem / totalMem) * 100);
  const cpu = getCpuUsage();
  return { cpu, mem: memUsage };
});

ipcMain.handle('read-file', async (event, filePath) => {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    return { success: true, content, path: filePath };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('write-file', async (event, data) => {
  try {
    fs.writeFileSync(data.path, data.content, 'utf-8');
    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('list-directory', async (event, dirPath) => {
  try {
    const files = fs.readdirSync(dirPath);
    const stats = files.map(file => {
      const fullPath = path.join(dirPath, file);
      let isDirectory = false;
      let size = 0;
      try {
        const stat = fs.statSync(fullPath);
        isDirectory = stat.isDirectory();
        size = stat.size || 0;
      } catch (e) {}
      return { name: file, isDirectory, size, modified: new Date() };
    });
    return { success: true, files: stats, path: dirPath };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('select-file', async (event, options) => {
  const result = await dialog.showOpenDialog(win, {
    properties: ['openFile'],
    filters: (options && options.filters) || []
  });
  if (result.canceled || result.filePaths.length === 0) return { filePath: null };
  return { filePath: result.filePaths[0] };
});

ipcMain.handle('self-correct', async (event, data) => {
  try {
    let correctedCode = data.code || data.originalContent || '';
    if (data.fixes && data.fixes.length > 0) {
      for (const fix of data.fixes) {
        if (fix.type === 'replace') {
          correctedCode = correctedCode.replace(fix.pattern, fix.replacement);
        } else if (fix.type === 'insert') {
          const lines = correctedCode.split('\n');
          lines.splice(fix.line, 0, fix.content);
          correctedCode = lines.join('\n');
        } else if (fix.type === 'delete') {
          const lines = correctedCode.split('\n');
          lines.splice(fix.line, 1);
          correctedCode = lines.join('\n');
        }
      }
    }
    if (data.filePath) {
      fs.writeFileSync(data.filePath, correctedCode, 'utf-8');
    }
    return {
      success: true,
      message: 'Self-correction applied successfully',
      corrected: correctedCode,
      filePath: data.filePath,
      issues: data.issues || [],
      fixes: data.fixes || []
    };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('list-workflows', async () => {
  return workflowStorage.loadJobs();
});

ipcMain.handle('schedule-workflow', async (event, jobData) => {
  try {
    const jobs = workflowStorage.loadJobs();
    const newJob = {
      id: Date.now().toString(36) + Math.random().toString(36).slice(2, 7),
      name: jobData.name,
      cronExpression: jobData.cronExpression,
      workflow: jobData.workflow,
      enabled: true
    };
    jobs.push(newJob);
    workflowStorage.saveJobs(jobs);
    workflowScheduler.scheduleJob(newJob, executeWorkflow);
    console.log('Workflow scheduled:', newJob.name);
    return newJob;
  } catch (error) {
    return { error: error.message };
  }
});

ipcMain.handle('unschedule-workflow', async (event, id) => {
  try {
    let jobs = workflowStorage.loadJobs();
    jobs = jobs.filter(job => job.id !== id);
    workflowStorage.saveJobs(jobs);
    workflowScheduler.unscheduleJob(id);
    return { success: true };
  } catch (error) {
    return { error: error.message };
  }
});

ipcMain.handle('run-workflow-now', async (event, id) => {
  try {
    const jobs = workflowStorage.loadJobs();
    const job = jobs.find(j => j.id === id);
    if (!job) throw new Error('Job not found');
    executeWorkflow(job);
    return { success: true };
  } catch (error) {
    return { error: error.message };
  }
});

ipcMain.handle('list-tools', async () => {
  try {
    return toolsStorage.loadTools();
  } catch (error) {
    return [];
  }
});

ipcMain.handle('propose-tool', async (event, toolDef) => {
  if (!toolDef || !toolDef.name || typeof toolDef.handlerCode !== 'string') {
    return { success: false, error: 'Tool definition is missing a name or handlerCode.' };
  }
  return new Promise((resolve) => {
    const pendingTool = {
      id: Date.now().toString(36) + Math.random().toString(36).slice(2, 7),
      name: String(toolDef.name).slice(0, 100),
      description: String(toolDef.description || '').slice(0, 1000),
      parameters: toolDef.parameters && typeof toolDef.parameters === 'object' ? toolDef.parameters : {},
      handlerCode: String(toolDef.handlerCode).slice(0, 20000),
      createdAt: new Date().toISOString()
    };

    if (!win || win.isDestroyed()) {
      resolve({ success: false, error: 'No window available to request authorization.' });
      return;
    }

    pendingToolConfirmations.set(pendingTool.id, (approved) => {
      if (!approved) {
        resolve({ success: false, declined: true });
        return;
      }
      try {
        const tools = toolsStorage.loadTools();
        tools.push(pendingTool);
        toolsStorage.saveTools(tools);
        resolve({ success: true, tool: pendingTool });
      } catch (err) {
        resolve({ success: false, error: err.message });
      }
    });

    safeSend('tool-creation-request', pendingTool);
  });
});

ipcMain.handle('run-tool', async (event, { name, params } = {}) => {
  try {
    const tools = toolsStorage.loadTools();
    const tool = tools.find((t) => t.name === name || t.id === name);
    if (!tool) return { success: false, error: `Tool "${name}" not found.` };
    return await toolRunner.executeTool(tool, params);
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('delete-tool', async (event, idOrName) => {
  try {
    const tools = toolsStorage.loadTools();
    const remaining = tools.filter((t) => t.id !== idOrName && t.name !== idOrName);
    const removed = remaining.length < tools.length;
    toolsStorage.saveTools(remaining);
    return { success: removed };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

function makeAIServerRequest(endpoint, method, data = null, timeout = 60000) {
  return new Promise((resolve, reject) => {
    let postData = null;
    if (data) {
      try {
        postData = JSON.stringify(data);
      } catch (e) {
        reject(new Error(`Failed to stringify data: ${e.message}`));
        return;
      }
    }

    const options = {
      hostname: aiServerHost,
      port: aiServerPort,
      path: endpoint,
      method: method,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      }
    };

    if (postData) {
      options.headers['Content-Length'] = Buffer.byteLength(postData);
    }

    const req = http.request(options, (res) => {
      let responseData = '';
      res.on('data', chunk => responseData += chunk);
      res.on('end', () => {
        try {
          if (!responseData || responseData.trim() === '') {
            resolve({ success: false, error: 'Empty response' });
            return;
          }
          const json = JSON.parse(responseData);
          resolve(json);
        } catch (e) {
          reject(new Error(`Failed to parse response: ${e.message}`));
        }
      });
    });

    req.on('error', error => reject(new Error(`Request failed: ${error.message}`)));
    if (timeout > 0) {
      req.setTimeout(timeout, () => {
        req.destroy();
        reject(new Error(`Request timed out after ${timeout}ms`));
      });
    }
    if (postData) req.write(postData);
    req.end();
  });
}

// ==============================================================
// VIDEO VIEW
// ==============================================================
function closeVideoView() {
  if (!videoView) return;
  try { if (win && !win.isDestroyed()) win.removeBrowserView(videoView); } catch (e) { /* ignore */ }
  videoView = null;
  isVideoOpen = false;
  console.log('🎬 Video view closed');
}

ipcMain.on('open-video', (event, url) => {
  if (!win || win.isDestroyed()) return;
  closeVideoView();

  videoView = new BrowserView({ webPreferences: { nodeIntegration: false, contextIsolation: true, plugins: true } });
  const viewWidth = Math.round(win.getBounds().width * 0.75);
  const viewHeight = Math.round(win.getBounds().height * 0.75);
  const x = Math.round((win.getBounds().width - viewWidth) / 2);
  const y = Math.round((win.getBounds().height - viewHeight) / 2);
  videoView.setBounds({ x, y, width: viewWidth, height: viewHeight });
  win.addBrowserView(videoView);
  isVideoOpen = true;

  // Scoped Escape: only fires when the video view itself has focus.
  videoView.webContents.on('before-input-event', (e, input) => {
    if (input.key === 'Escape' && input.type === 'keyDown') {
      e.preventDefault();
      closeVideoView();
    }
  });

  const finalUrl = url.includes('?') ? url + '&playsinline=1' : url + '?playsinline=1';
  videoView.webContents.loadURL(finalUrl);
  console.log('🎬 Video view opened: ' + finalUrl);
});

ipcMain.on('close-video', () => closeVideoView());

// ==============================================================
// MAP VIEW HANDLERS
// ==============================================================

function closeMapView() {
  if (!mapView) return;
  try { if (win && !win.isDestroyed()) win.removeBrowserView(mapView); } catch (e) { /* ignore */ }
  mapView = null;
  isMapOpen = false;
  console.log('🗺️ Map view closed');
}

ipcMain.on('open-map', (event, mapData) => {
  if (!win || win.isDestroyed()) return;

  closeMapView();

  var mapHtml = generateMapHtml(mapData || { center: [0, 0], zoom: 2 });

  mapView = new BrowserView({
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: true
    }
  });

  var viewWidth = Math.round(win.getBounds().width * 0.85);
  var viewHeight = Math.round(win.getBounds().height * 0.80);
  var x = Math.round((win.getBounds().width - viewWidth) / 2);
  var y = Math.round((win.getBounds().height - viewHeight) / 2);

  mapView.setBounds({ x: x, y: y, width: viewWidth, height: viewHeight });
  win.addBrowserView(mapView);
  isMapOpen = true;

  const view = mapView;

  // Escape closes the map — mirrors video/browser behavior.
  view.webContents.on('before-input-event', (e, input) => {
    if (input.key === 'Escape' && input.type === 'keyDown') {
      e.preventDefault();
      if (mapView === view) closeMapView();
    }
  });

  // The injected closeMap() logs this marker; it's the ONLY reliable path
  // for the in-page X button to signal the main process from a BrowserView.
  view.webContents.on('console-message', (e, level, message) => {
    if (typeof message === 'string' && message.trim() === 'JARVIS_CLOSE_MAP') {
      if (mapView === view) closeMapView();
    }
  });

  view.webContents.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(mapHtml));
  console.log('🗺️ Map view opened');
});

ipcMain.on('close-map', () => closeMapView());

function generateMapHtml(mapData) {
  const center = mapData.center || [0, 0];
  const zoom = mapData.zoom || 2;
  const markers = mapData.markers || [];

  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>J.A.R.V.I.S. - Offline Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background: #0a0a12;
      font-family: "Rajdhani", sans-serif;
      overflow: hidden;
    }
    #map {
      height: 100vh;
      width: 100vw;
      background: #0a0a12;
    }
    .map-controls {
      position: absolute;
      top: 20px;
      right: 20px;
      z-index: 1000;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .map-controls button {
      background: rgba(10, 10, 18, 0.85);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(0, 200, 255, 0.2);
      color: #00e5ff;
      padding: 8px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-family: "Rajdhani", sans-serif;
      font-size: 13px;
      font-weight: 600;
      transition: all 0.2s;
      text-shadow: 0 0 10px rgba(0,229,255,0.2);
    }
    .map-controls button:hover {
      background: rgba(0, 200, 255, 0.15);
      border-color: rgba(0, 200, 255, 0.4);
      box-shadow: 0 0 20px rgba(0, 200, 255, 0.1);
    }
    .map-controls .close-btn {
      color: #ff6b6b;
      border-color: rgba(255, 50, 50, 0.2);
    }
    .map-controls .close-btn:hover {
      background: rgba(255, 50, 50, 0.15);
      border-color: rgba(255, 50, 50, 0.4);
    }
    .map-title {
      position: absolute;
      top: 20px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 1000;
      color: rgba(255, 255, 255, 0.3);
      font-size: 12px;
      letter-spacing: 4px;
      text-transform: uppercase;
      text-shadow: 0 0 20px rgba(0,200,255,0.1);
      pointer-events: none;
    }
    .leaflet-control-attribution {
      background: rgba(10, 10, 18, 0.8) !important;
      color: rgba(255, 255, 255, 0.3) !important;
      font-size: 9px !important;
      backdrop-filter: blur(4px);
    }
    .leaflet-control-attribution a {
      color: rgba(0, 200, 255, 0.5) !important;
    }
    .leaflet-popup-content-wrapper {
      background: rgba(10, 10, 18, 0.9) !important;
      backdrop-filter: blur(8px);
      border: 1px solid rgba(0, 200, 255, 0.1);
      color: #fff !important;
      border-radius: 8px !important;
    }
    .leaflet-popup-tip {
      background: rgba(10, 10, 18, 0.9) !important;
    }
    .loading-overlay {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(10, 10, 18, 0.9);
      z-index: 999;
      color: rgba(255, 255, 255, 0.3);
      font-size: 14px;
      letter-spacing: 2px;
    }
    .loading-overlay .spinner {
      width: 40px;
      height: 40px;
      border: 2px solid rgba(0, 200, 255, 0.1);
      border-top-color: #00e5ff;
      border-radius: 50%;
      animation: spin 1s linear infinite;
      margin-right: 16px;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    .search-box {
      position: absolute;
      bottom: 30px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 1000;
      display: flex;
      gap: 8px;
      background: rgba(10, 10, 18, 0.85);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(0, 200, 255, 0.1);
      border-radius: 8px;
      padding: 8px 12px;
    }
    .search-box input {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 4px;
      padding: 6px 12px;
      color: #fff;
      font-family: "JetBrains Mono", monospace;
      font-size: 12px;
      width: 200px;
      outline: none;
    }
    .search-box input:focus {
      border-color: rgba(0, 200, 255, 0.3);
    }
    .search-box button {
      background: rgba(0, 200, 255, 0.15);
      border: 1px solid rgba(0, 200, 255, 0.2);
      border-radius: 4px;
      padding: 6px 16px;
      color: #00e5ff;
      font-family: "Rajdhani", sans-serif;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    .search-box button:hover {
      background: rgba(0, 200, 255, 0.25);
    }
    .search-results {
      position: absolute;
      bottom: 80px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 1000;
      background: rgba(10, 10, 18, 0.9);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(0, 200, 255, 0.1);
      border-radius: 8px;
      padding: 8px;
      max-height: 200px;
      overflow-y: auto;
      min-width: 300px;
      display: none;
    }
    .search-results .result-item {
      padding: 6px 12px;
      color: rgba(255, 255, 255, 0.6);
      cursor: pointer;
      border-radius: 4px;
      font-size: 13px;
      transition: all 0.2s;
    }
    .search-results .result-item:hover {
      background: rgba(0, 200, 255, 0.1);
      color: #fff;
    }
  </style>
</head>
<body>
  <div class="loading-overlay" id="loading">
    <div class="spinner"></div>
    <span>LOADING MAP DATA...</span>
  </div>

  <div class="map-title">J.A.R.V.I.S. • GLOBAL POSITIONING SYSTEM</div>

  <div class="map-controls">
    <button onclick="centerMap()">🎯 CENTER</button>
    <button onclick="toggleSatellite()">🛰️ SATELLITE</button>
    <button onclick="resetView()">⟲ RESET</button>
    <button onclick="toggleMarkers()">📍 MARKERS</button>
    <button class="close-btn" onclick="closeMap()">✕ CLOSE</button>
  </div>

  <div class="search-box">
    <input type="text" id="searchInput" placeholder="Search location..." />
    <button onclick="searchLocation()">🔍</button>
  </div>
  <div class="search-results" id="searchResults"></div>

  <div id="map"></div>

  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

  <script>
    var map;
    var isSatellite = false;
    var currentLayer;
    var markers = [];
    var markerLayer;
    var showMarkers = true;
    var mapInitialized = false;

    var mapData = ${JSON.stringify(mapData)};
    var markersData = ${JSON.stringify(markers)};

    var worldCities = {
      "london": { lat: 51.5074, lng: -0.1278, country: "UK" },
      "paris": { lat: 48.8566, lng: 2.3522, country: "France" },
      "new york": { lat: 40.7128, lng: -74.0060, country: "USA" },
      "tokyo": { lat: 35.6762, lng: 139.6503, country: "Japan" },
      "sydney": { lat: -33.8688, lng: 151.2093, country: "Australia" },
      "mumbai": { lat: 19.0760, lng: 72.8777, country: "India" },
      "cairo": { lat: 30.0444, lng: 31.2357, country: "Egypt" },
      "moscow": { lat: 55.7558, lng: 37.6173, country: "Russia" },
      "beijing": { lat: 39.9042, lng: 116.4074, country: "China" },
      "rio": { lat: -22.9068, lng: -43.1729, country: "Brazil" },
      "cape town": { lat: -33.9249, lng: 18.4241, country: "South Africa" },
      "dubai": { lat: 25.2048, lng: 55.2708, country: "UAE" },
      "los angeles": { lat: 34.0522, lng: -118.2437, country: "USA" },
      "chicago": { lat: 41.8781, lng: -87.6298, country: "USA" },
      "toronto": { lat: 43.6532, lng: -79.3832, country: "Canada" },
      "berlin": { lat: 52.5200, lng: 13.4050, country: "Germany" },
      "rome": { lat: 41.9028, lng: 12.4964, country: "Italy" },
      "madrid": { lat: 40.4168, lng: -3.7038, country: "Spain" },
      "amsterdam": { lat: 52.3676, lng: 4.9041, country: "Netherlands" },
      "bangkok": { lat: 13.7563, lng: 100.5018, country: "Thailand" },
      "singapore": { lat: 1.3521, lng: 103.8198, country: "Singapore" },
      "hong kong": { lat: 22.3193, lng: 114.1694, country: "China" },
      "seoul": { lat: 37.5665, lng: 126.9780, country: "South Korea" },
      "istanbul": { lat: 41.0082, lng: 28.9784, country: "Turkey" },
      "mexico city": { lat: 19.4326, lng: -99.1332, country: "Mexico" }
    };

    function initMap() {
      map = L.map("map", {
        center: [${center[0]}, ${center[1]}],
        zoom: ${zoom},
        zoomControl: true,
        fadeAnimation: true,
        zoomAnimation: true,
        wheelPxPerZoomLevel: 120
      });

      currentLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "© OpenStreetMap contributors",
        maxZoom: 19,
        minZoom: 1
      }).addTo(map);

      markerLayer = L.layerGroup().addTo(map);

      markersData.forEach(function(markerData) {
        var marker = L.marker([markerData.lat, markerData.lng]);
        if (markerData.popup) {
          marker.bindPopup(markerData.popup);
        }
        marker.addTo(markerLayer);
        markers.push(marker);
      });

      document.getElementById("loading").style.display = "none";
      mapInitialized = true;
      console.log("🗺️ Map initialized with", markers.length, "markers");
    }

    function centerMap() {
      if (!map) return;
      map.setView([${center[0]}, ${center[1]}], ${zoom});
    }

    function resetView() {
      if (!map) return;
      map.setView([${center[0]}, ${center[1]}], ${zoom});
      if (isSatellite) toggleSatellite();
      if (!showMarkers) toggleMarkers();
    }

    function toggleSatellite() {
      if (!map) return;
      isSatellite = !isSatellite;
      map.removeLayer(currentLayer);
      if (isSatellite) {
        currentLayer = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
          attribution: "© Esri",
          maxZoom: 19,
          minZoom: 1
        }).addTo(map);
        console.log("🛰️ Satellite view enabled");
      } else {
        currentLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
          attribution: "© OpenStreetMap contributors",
          maxZoom: 19,
          minZoom: 1
        }).addTo(map);
        console.log("🗺️ Standard view enabled");
      }
    }

    function toggleMarkers() {
      if (!map) return;
      showMarkers = !showMarkers;
      if (showMarkers) markerLayer.addTo(map);
      else map.removeLayer(markerLayer);
      console.log("📍 Markers:", showMarkers ? "shown" : "hidden");
    }

    function closeMap() {
      try { console.log("JARVIS_CLOSE_MAP"); } catch (e) {}
      try {
        if (window.parent && window.parent.postMessage) {
          window.parent.postMessage("close-map", "*");
        }
      } catch(e) {}
      try {
        if (window.electronAPI && window.electronAPI.closeMap) {
          window.electronAPI.closeMap();
        }
      } catch(e) {}
      try { if (window.close) window.close(); } catch(e) {}
      try {
        if (window.location && window.location.hash !== "#close") {
          window.location.hash = "close";
        }
      } catch(e) {}
    }

    window.addEventListener("message", function(event) {
      if (event.data === "close-map") {
        try { console.log("JARVIS_CLOSE_MAP"); } catch (e) {}
      }
    });

    window.addEventListener("hashchange", function() {
      if (window.location.hash === "#close") {
        try { console.log("JARVIS_CLOSE_MAP"); } catch (e) {}
      }
    });

    function searchLocation() {
      var input = document.getElementById("searchInput");
      var query = input.value.toLowerCase().trim();
      var results = document.getElementById("searchResults");

      if (!query) {
        results.style.display = "none";
        return;
      }

      var matches = [];
      for (var name in worldCities) {
        if (name.indexOf(query) !== -1) {
          matches.push({ name: name, lat: worldCities[name].lat, lng: worldCities[name].lng, country: worldCities[name].country });
        }
      }

      if (matches.length === 0) {
        results.innerHTML = '<div class="result-item">No results found</div>';
        results.style.display = "block";
        return;
      }

      var html = "";
      for (var i = 0; i < matches.length; i++) {
        var match = matches[i];
        var displayName = match.name.charAt(0).toUpperCase() + match.name.slice(1);
        html += '<div class="result-item" onclick="goToLocation(' + match.lat + ', ' + match.lng + ', ' + JSON.stringify(match.name) + ')">' + displayName + ', ' + match.country + '</div>';
      }
      results.innerHTML = html;
      results.style.display = "block";
    }

    function goToLocation(lat, lng, name) {
      if (!map) return;
      map.setView([lat, lng], 12);
      document.getElementById("searchResults").style.display = "none";
      document.getElementById("searchInput").value = "";

      var marker = L.marker([lat, lng]).addTo(map);
      marker.bindPopup("📍 " + name).openPopup();

      setTimeout(function() {
        map.removeLayer(marker);
      }, 5000);
    }

    document.addEventListener("keydown", function(e) {
      if (e.key === "Escape") { closeMap(); }
      if (e.key === "c" || e.key === "C") { centerMap(); }
      if (e.key === "s" || e.key === "S") { toggleSatellite(); }
      if (e.key === "r" || e.key === "R") { resetView(); }
      if (e.key === "m" || e.key === "M") { toggleMarkers(); }
      if (e.key === "Enter") { searchLocation(); }
    });

    document.getElementById("searchInput").addEventListener("input", searchLocation);
    document.getElementById("searchInput").addEventListener("blur", function() {
      setTimeout(function() {
        document.getElementById("searchResults").style.display = "none";
      }, 200);
    });
    document.getElementById("searchInput").addEventListener("focus", function() {
      if (document.getElementById("searchInput").value) searchLocation();
    });

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", initMap);
    } else {
      initMap();
    }

    window.addEventListener("resize", function() {
      if (map) setTimeout(function() { map.invalidateSize(); }, 100);
    });
  </script>
</body>
</html>`;
}

// ==============================================================
// SKY VIEW
// ==============================================================
function closeSkyView() {
  if (!skyView) return;
  try { if (win && !win.isDestroyed()) win.removeBrowserView(skyView); } catch (e) { /* ignore */ }
  skyView = null;
  isSkyOpen = false;
  safeSend('sky-closed');
  console.log('🌌 Sky view closed');
}

async function openSkyView(payload) {
  if (!win || win.isDestroyed()) return { success: false, error: 'No window available' };
  closeSkyView();

  const req = {
    lat: Number(payload && payload.lat) || 5.65,
    lng: Number(payload && payload.lng) || 100.49,
    time: (payload && payload.time) || null,
    name: (payload && payload.name) || '',
  };

  let positions;
  try {
    positions = await makeAIServerRequest('/sky/positions', 'POST', req, 20000);
  } catch (err) {
    console.error('🌌 Sky positions request failed:', err.message);
    safeSend('sky-error', err.message);
    return { success: false, error: err.message };
  }
  if (!positions || positions.success === false) {
    const msg = (positions && positions.error) || 'Failed to load sky positions';
    console.error('🌌 Sky positions error:', msg);
    safeSend('sky-error', msg);
    return { success: false, error: msg };
  }

  const html = generateSkyHtml(positions);
  skyView = new BrowserView({
    webPreferences: { nodeIntegration: false, contextIsolation: true }
  });
  const viewWidth = Math.round(win.getBounds().width * 0.88);
  const viewHeight = Math.round(win.getBounds().height * 0.86);
  const x = Math.round((win.getBounds().width - viewWidth) / 2);
  const y = Math.round((win.getBounds().height - viewHeight) / 2);
  skyView.setBounds({ x, y, width: viewWidth, height: viewHeight });
  win.addBrowserView(skyView);
  isSkyOpen = true;

  const view = skyView;
  view.webContents.on('before-input-event', (e, input) => {
    if (input.key === 'Escape' && input.type === 'keyDown') {
      e.preventDefault();
      if (skyView === view) closeSkyView();
    }
  });
  view.webContents.on('console-message', (event, _level, _message) => {
    const text = (event && typeof event === 'object' && typeof event.message === 'string')
      ? event.message
      : _message;
    if (typeof text === 'string' && text.trim() === 'JARVIS_CLOSE_SKY') {
      if (skyView === view) closeSkyView();
    }
  });

  view.webContents.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(html));
  safeSend('sky-opened', { mode: positions.mode, observer: positions.observer });
  console.log('🌌 Sky view opened (' + positions.mode + ')');
  return { success: true, mode: positions.mode };
}

ipcMain.on('open-sky', (event, payload) => {
  openSkyView(payload || {}).catch((err) => {
    console.error('🌌 openSkyView failed:', err);
    safeSend('sky-error', err.message);
  });
});

ipcMain.on('close-sky', () => closeSkyView());

async function bridgeOpenSky(args) {
  return await openSkyView(args || {});
}

function generateSkyHtml(positionsData) {
  const mode = positionsData.mode || 'demo';
  const observer = positionsData.observer || { name: 'Unknown', lat: 0, lng: 0 };
  const timeUtc = positionsData.time_utc || '';

  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>J.A.R.V.I.S. — Sky View</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: #050510; overflow: hidden; font-family: 'Rajdhani', system-ui, sans-serif; color: #fff; }
    #sky { display: block; width: 100vw; height: 100vh; cursor: grab; touch-action: none; }
    #sky.dragging { cursor: grabbing; }
    .overlay {
      position: absolute; top: 20px; left: 20px; padding: 14px 18px;
      background: rgba(10,10,18,0.85); border: 1px solid rgba(0,200,255,0.25);
      border-radius: 8px; font-family: monospace; font-size: 12px;
      backdrop-filter: blur(8px); max-width: 340px;
    }
    .overlay h1 { font-size: 14px; letter-spacing: 3px; color: #00e5ff; margin-bottom: 6px; font-family: 'Rajdhani', sans-serif; }
    .overlay .row { color: rgba(255,255,255,0.6); padding: 2px 0; }
    .overlay .row b { color: rgba(255,255,255,0.85); }
    .mode-badge { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 10px; letter-spacing: 1px; margin-left: 6px; }
    .mode-live { background: rgba(0,255,136,0.15); color: #00ff88; border: 1px solid rgba(0,255,136,0.3); }
    .mode-demo { background: rgba(255,170,0,0.15); color: #ffaa00; border: 1px solid rgba(255,170,0,0.3); }
    .controls { position: absolute; top: 20px; right: 20px; display: flex; flex-direction: column; gap: 8px; z-index: 10; }
    .controls button {
      padding: 8px 16px; background: rgba(10,10,18,0.85); backdrop-filter: blur(8px);
      border: 1px solid rgba(0,200,255,0.25); color: #00e5ff; border-radius: 6px;
      cursor: pointer; font-family: 'Rajdhani', sans-serif; font-size: 13px; font-weight: 600; letter-spacing: 1px;
    }
    .controls button:hover { background: rgba(0,200,255,0.15); }
    .controls .close-btn { color: #ff6b6b; border-color: rgba(255,50,50,0.3); }
    .controls .close-btn:hover { background: rgba(255,50,50,0.15); }
    .legend {
      position: absolute; bottom: 20px; left: 20px; padding: 12px 16px;
      background: rgba(10,10,18,0.85); backdrop-filter: blur(8px);
      border: 1px solid rgba(0,200,255,0.15); border-radius: 8px; font-size: 12px;
      font-family: monospace; max-height: 40vh; overflow-y: auto;
    }
    .legend .item { display: flex; align-items: center; gap: 10px; padding: 3px 0; }
    .legend .dot { width: 10px; height: 10px; border-radius: 50%; box-shadow: 0 0 8px currentColor; }
    .legend .dim { opacity: 0.35; }
    .legend .info { color: rgba(255,255,255,0.5); font-size: 10px; margin-left: auto; }
    .hint { position: absolute; bottom: 20px; right: 20px; padding: 10px 14px;
            background: rgba(10,10,18,0.85); border: 1px solid rgba(0,200,255,0.15);
            border-radius: 6px; font-size: 11px; color: rgba(255,255,255,0.4); font-family: monospace; }
  </style>
</head>
<body>
  <canvas id="sky"></canvas>

  <div class="overlay">
    <h1>SKY VIEW <span class="mode-badge ${mode === 'live' ? 'mode-live' : 'mode-demo'}">${mode === 'live' ? 'LIVE' : 'DEMO'}</span></h1>
    <div class="row"><b>Observer:</b> ${observer.name || '—'}</div>
    <div class="row"><b>Lat/Lng:</b> ${Number(observer.lat).toFixed(3)}, ${Number(observer.lng).toFixed(3)}</div>
    <div class="row"><b>Time (UTC):</b> ${timeUtc}</div>
    <div class="row"><b>Facing:</b> <span id="facingReadout">N (0°)</span></div>
  </div>

  <div class="controls">
    <button onclick="resetView()">⟲ RESET</button>
    <button onclick="toggleLabels()">LABELS</button>
    <button onclick="toggleStars()">STARS</button>
    <button class="close-btn" onclick="closeSky()">✕ CLOSE</button>
  </div>

  <div class="legend" id="legend"></div>
  <div class="hint">Drag to rotate • Wheel to zoom • Esc to close</div>

<script>
  const DATA = ${JSON.stringify(positionsData)};

  const canvas = document.getElementById('sky');
  const ctx = canvas.getContext('2d');
  let showLabels = true;
  let showStars = true;

  // View state — mutable, driven by drag and wheel.
  let azOffset = 0;       // degrees; 0 = North at top
  let radiusScale = 1.0;  // 1.0 = default dome size

  // Drag state.
  let isDragging = false;
  let lastMouseX = 0;

  const STAR_SEED = 0xC0FFEE;
  let _s = STAR_SEED >>> 0;
  function rand() { _s = (_s * 1664525 + 1013904223) >>> 0; return _s / 0xFFFFFFFF; }
  const STARS = [];
  for (let i = 0; i < 500; i++) {
    STARS.push({
      az: rand() * 360,
      alt: Math.pow(rand(), 0.6) * 90,
      mag: rand() * 6,
      tw: rand() * Math.PI * 2,
    });
  }

  function setupCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const w = window.innerWidth, h = window.innerHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w, h };
  }

  function project(alt, az, cx, cy, R) {
    const r = R * (90 - alt) / 90;
    const a = ((az + azOffset) * Math.PI / 180) - Math.PI / 2;
    return { x: cx + Math.cos(a) * r, y: cy + Math.sin(a) * r };
  }

  function normalizeDeg(d) {
    d = d % 360;
    if (d < 0) d += 360;
    return d;
  }

  function updateFacingReadout() {
    const facing = normalizeDeg(azOffset);
    // The direction currently at the TOP of the dome is the direction
    // whose azimuth equals -azOffset in the unrotated frame.
    const topAz = normalizeDeg(-facing);
    const dirs = [
      { label: 'N', az: 0 },
      { label: 'NE', az: 45 },
      { label: 'E', az: 90 },
      { label: 'SE', az: 135 },
      { label: 'S', az: 180 },
      { label: 'SW', az: 225 },
      { label: 'W', az: 270 },
      { label: 'NW', az: 315 },
    ];
    let best = dirs[0];
    let bestDist = 999;
    for (const d of dirs) {
      const diff = Math.min(
        Math.abs(d.az - topAz),
        360 - Math.abs(d.az - topAz)
      );
      if (diff < bestDist) { bestDist = diff; best = d; }
    }
    const el = document.getElementById('facingReadout');
    if (el) {
      el.textContent = best.label + ' (' + Math.round(topAz) + '\\u00B0)';
    }
  }

  function draw(t) {
    const { w, h } = setupCanvas();
    const cx = w / 2, cy = h / 2;
    const R = Math.min(w, h) * 0.40 * radiusScale;

    ctx.fillStyle = '#050510';
    ctx.fillRect(0, 0, w, h);

    const sky = ctx.createRadialGradient(cx, cy, 0, cx, cy, R);
    sky.addColorStop(0,    'rgba(25, 35, 70, 0.95)');
    sky.addColorStop(0.55, 'rgba(15, 20, 45, 0.85)');
    sky.addColorStop(1,    'rgba(5, 8, 20, 0.75)');
    ctx.fillStyle = sky;
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.fill();

    if (showStars) {
      for (const st of STARS) {
        const p = project(st.alt, st.az, cx, cy, R);
        if (p.x < 0 || p.x > w || p.y < 0 || p.y > h) continue;
        const twinkle = 0.7 + 0.3 * Math.sin(t / 800 + st.tw);
        const size = Math.max(0.4, (6 - st.mag) * 0.35);
        ctx.fillStyle = 'rgba(255,255,255,' + (twinkle * (0.9 - st.mag * 0.1)) + ')';
        ctx.beginPath();
        ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    for (const alt of [30, 60]) {
      const r = R * (90 - alt) / 90;
      ctx.strokeStyle = 'rgba(0,200,255,0.12)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 5]);
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    ctx.strokeStyle = 'rgba(0,200,255,0.45)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.stroke();

    const cardinals = [ ['N', 0], ['E', 90], ['S', 180], ['W', 270] ];
    ctx.font = 'bold 18px system-ui';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const [name, az] of cardinals) {
      const a = ((az + azOffset) * Math.PI / 180) - Math.PI / 2;
      const x = cx + Math.cos(a) * (R + 28);
      const y = cy + Math.sin(a) * (R + 28);
      ctx.fillStyle = 'rgba(0,229,255,0.75)';
      ctx.fillText(name, x, y);
    }

    ctx.strokeStyle = 'rgba(0,200,255,0.2)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 6, cy); ctx.lineTo(cx + 6, cy);
    ctx.moveTo(cx, cy - 6); ctx.lineTo(cx, cy + 6);
    ctx.stroke();

    for (const body of DATA.bodies) {
      if (!body.visible) continue;
      const p = project(body.altitude, body.azimuth, cx, cy, R);
      const size = Math.max(4, Math.min(18, 12 - body.magnitude));
      ctx.save();
      ctx.shadowBlur = 22;
      ctx.shadowColor = body.color;
      ctx.fillStyle = body.color;
      ctx.beginPath();
      ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      ctx.fillStyle = 'rgba(255,255,255,0.6)';
      ctx.beginPath();
      ctx.arc(p.x - size * 0.25, p.y - size * 0.25, size * 0.35, 0, Math.PI * 2);
      ctx.fill();

      if (showLabels) {
        ctx.fillStyle = 'rgba(255,255,255,0.9)';
        ctx.font = '13px system-ui';
        ctx.textAlign = 'center';
        ctx.fillText(body.name, p.x, p.y - size - 8);
        ctx.fillStyle = 'rgba(255,255,255,0.45)';
        ctx.font = '10px monospace';
        ctx.fillText(body.altitude.toFixed(0) + '\\u00B0 alt', p.x, p.y - size - 20);
      }
    }

    requestAnimationFrame(draw);
  }

  // -------- Drag to rotate --------
  function onDown(clientX) {
    isDragging = true;
    lastMouseX = clientX;
    canvas.classList.add('dragging');
  }
  function onMove(clientX) {
    if (!isDragging) return;
    const dx = clientX - lastMouseX;
    lastMouseX = clientX;
    azOffset += dx * 0.4;
    azOffset = normalizeDeg(azOffset);
    updateFacingReadout();
  }
  function onUp() {
    isDragging = false;
    canvas.classList.remove('dragging');
  }

  canvas.addEventListener('mousedown', (e) => { e.preventDefault(); onDown(e.clientX); });
  canvas.addEventListener('mousemove', (e) => onMove(e.clientX));
  canvas.addEventListener('mouseup', onUp);
  canvas.addEventListener('mouseleave', onUp);

  canvas.addEventListener('touchstart', (e) => {
    if (e.touches.length === 1) onDown(e.touches[0].clientX);
  }, { passive: true });
  canvas.addEventListener('touchmove', (e) => {
    if (e.touches.length === 1) onMove(e.touches[0].clientX);
  }, { passive: true });
  canvas.addEventListener('touchend', onUp);
  canvas.addEventListener('touchcancel', onUp);

  // -------- Wheel to zoom --------
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    if (e.deltaY < 0) radiusScale *= 1.08;
    else radiusScale *= 0.92;
    radiusScale = Math.max(0.4, Math.min(2.5, radiusScale));
  }, { passive: false });

  // -------- Buttons --------
  window.resetView = function() {
    azOffset = 0;
    radiusScale = 1.0;
    updateFacingReadout();
  };
  window.toggleLabels = function() { showLabels = !showLabels; };
  window.toggleStars = function() { showStars = !showStars; };
  window.closeSky = function() {
    try { console.log('JARVIS_CLOSE_SKY'); } catch (e) {}
  };

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') window.closeSky();
    if (e.key === 'l' || e.key === 'L') window.toggleLabels();
    if (e.key === 's' || e.key === 'S') window.toggleStars();
    if (e.key === 'r' || e.key === 'R') window.resetView();
  });

  (function buildLegend() {
    const el = document.getElementById('legend');
    let html = '';
    for (const body of DATA.bodies) {
      const dim = body.visible ? '' : ' dim';
      html += '<div class="item' + dim + '">' +
        '<span class="dot" style="background:' + body.color + '; color:' + body.color + '"></span>' +
        '<span>' + body.name + '</span>' +
        '<span class="info">' + (body.visible ? body.altitude.toFixed(0) + '\\u00B0 / ' + body.azimuth.toFixed(0) + '\\u00B0' : 'below horizon') + '</span>' +
        '</div>';
    }
    if (DATA.mode === 'demo') {
      html += '<div style="margin-top:8px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.1); color:#ffaa00; font-size:10px;">' +
              'Demo sky — install Skyfield for real positions (pip install skyfield)' +
              '</div>';
    }
    el.innerHTML = html;
  })();

  updateFacingReadout();
  requestAnimationFrame(draw);
</script>
</body>
</html>`;
}

// ==============================================================
// WEB BROWSER VIEW
// ==============================================================
function openBrowserView(url) {
  return new Promise((resolve, reject) => {
    if (!win || win.isDestroyed()) { reject(new Error('No window available')); return; }
    closeBrowserView(true);

    const view = new BrowserView({
      webPreferences: { nodeIntegration: false, contextIsolation: true }
    });
    browserView = view;
    isBrowserOpen = true;

    const viewWidth = Math.round(win.getBounds().width * 0.85);
    const viewHeight = Math.round(win.getBounds().height * 0.78);
    const x = Math.round((win.getBounds().width - viewWidth) / 2);
    const y = 40;
    view.setBounds({ x, y, width: viewWidth, height: viewHeight });
    win.addBrowserView(view);

    const startUrl = url || 'https://www.google.com';
    browserHistory = [startUrl];
    browserIndex = 0;

    view.webContents.on('before-input-event', (e, input) => {
      if (input.key === 'Escape' && input.type === 'keyDown') {
        e.preventDefault();
        if (browserView === view) closeBrowserView();
      }
    });

    const sendUrlChanged = () => {
      if (view.webContents.isDestroyed()) return;
      safeSend('browser-url-changed', {
        url: view.webContents.getURL(),
        title: view.webContents.getTitle(),
      });
    };
    view.webContents.on('did-navigate', sendUrlChanged);
    view.webContents.on('did-navigate-in-page', sendUrlChanged);

    let settled = false;
    const cleanup = () => {
      if (view.webContents.isDestroyed()) return;
      view.webContents.removeListener('did-finish-load', onFinish);
      view.webContents.removeListener('did-fail-load', onFail);
    };
    const onFinish = () => {
      if (settled) return;
      settled = true;
      cleanup();
      if (view.webContents.isDestroyed()) {
        reject(new Error('Browser view was destroyed before load finished'));
        return;
      }
      const result = { url: view.webContents.getURL(), title: view.webContents.getTitle() };
      safeSend('browser-opened', result);
      resolve(result);
    };
    const onFail = (event, errorCode, errorDescription) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error(`Failed to load (${errorCode}): ${errorDescription}`));
    };
    view.webContents.once('did-finish-load', onFinish);
    view.webContents.once('did-fail-load', onFail);

    view.webContents.loadURL(startUrl);
    console.log('🌐 Browser opened: ' + startUrl);
  });
}

ipcMain.on('open-browser', (event, url) => {
  openBrowserView(url).catch((err) => console.error('Browser open error:', err.message));
});

function closeBrowserView(silent) {
  if (browserView) {
    try { if (win && !win.isDestroyed()) win.removeBrowserView(browserView); } catch (e) {}
    browserView = null;
    if (!silent) safeSend('browser-closed');
  }
  isBrowserOpen = false;
  browserHistory = [];
  browserIndex = -1;
  console.log('🌐 Browser closed');
}

// ==============================================================
// BROWSER AUTOMATION ACTIONS
// ==============================================================
async function bridgeNavigate(args) {
  const rawUrl = String((args && args.url) || '').trim();
  if (!rawUrl) return { success: false, error: 'url is required' };
  let finalUrl = rawUrl;
  if (!finalUrl.startsWith('http://') && !finalUrl.startsWith('https://')) {
    finalUrl = (finalUrl.includes(' ') || !finalUrl.includes('.'))
      ? 'https://www.google.com/search?q=' + encodeURIComponent(finalUrl)
      : 'https://' + finalUrl;
  }
  try {
    const result = await openBrowserView(finalUrl);
    return { success: true, url: result.url, title: result.title };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

async function bridgeClick(args) {
  if (!browserView || browserView.webContents.isDestroyed()) {
    return { success: false, error: 'Browser is not open — call browser_navigate first.' };
  }
  const selector = args && args.selector;
  const text = args && args.text;
  if (!selector && !text) return { success: false, error: 'selector or text is required' };
  const script = `
    (function() {
      try {
        let el = null;
        ${selector ? `el = document.querySelector(${JSON.stringify(selector)});` : ''}
        ${text ? `
        if (!el) {
          const candidates = Array.from(document.querySelectorAll('a, button, [role="button"], input[type="submit"], input[type="button"], summary'));
          const needle = ${JSON.stringify(String(text).toLowerCase())};
          el = candidates.find(function(c) { return (c.innerText || c.value || '').trim().toLowerCase().includes(needle); });
        }` : ''}
        if (!el) return { success: false, error: 'Element not found' };
        el.scrollIntoView({ block: 'center' });
        el.click();
        return { success: true };
      } catch (e) {
        return { success: false, error: e.message };
      }
    })()
  `;
  try {
    return await browserView.webContents.executeJavaScript(script, true);
  } catch (err) {
    return { success: false, error: err.message };
  }
}

async function bridgeFill(args) {
  if (!browserView || browserView.webContents.isDestroyed()) {
    return { success: false, error: 'Browser is not open — call browser_navigate first.' };
  }
  const selector = args && args.selector;
  const value = args && args.value;
  if (!selector) return { success: false, error: 'selector is required' };
  if (value === undefined || value === null) return { success: false, error: 'value is required' };
  const script = `
    (function() {
      try {
        const el = document.querySelector(${JSON.stringify(selector)});
        if (!el) return { success: false, error: 'Element not found' };
        const proto = Object.getPrototypeOf(el);
        const desc = Object.getOwnPropertyDescriptor(proto, 'value');
        if (desc && desc.set) {
          desc.set.call(el, ${JSON.stringify(String(value))});
        } else {
          el.value = ${JSON.stringify(String(value))};
        }
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
        return { success: true };
      } catch (e) {
        return { success: false, error: e.message };
      }
    })()
  `;
  try {
    return await browserView.webContents.executeJavaScript(script, true);
  } catch (err) {
    return { success: false, error: err.message };
  }
}

async function bridgeRead(args) {
  if (!browserView || browserView.webContents.isDestroyed()) {
    return { success: false, error: 'Browser is not open — call browser_navigate first.' };
  }
  const selector = args && args.selector;
  const script = `
    (function() {
      try {
        const el = ${selector ? `document.querySelector(${JSON.stringify(selector)})` : 'document.body'};
        if (!el) return { success: false, error: 'Element not found' };
        return { success: true, text: (el.innerText || '').slice(0, 4000), url: location.href, title: document.title };
      } catch (e) {
        return { success: false, error: e.message };
      }
    })()
  `;
  try {
    return await browserView.webContents.executeJavaScript(script, true);
  } catch (err) {
    return { success: false, error: err.message };
  }
}

async function bridgeScreenshot() {
  if (!browserView || browserView.webContents.isDestroyed()) {
    return { success: false, error: 'Browser is not open — call browser_navigate first.' };
  }
  try {
    const image = await browserView.webContents.capturePage();
    const dir = path.join(app.getPath('userData'), 'screenshots');
    fs.mkdirSync(dir, { recursive: true });
    const filePath = path.join(dir, `screenshot-${Date.now()}.png`);
    fs.writeFileSync(filePath, image.toPNG());
    return {
      success: true,
      path: filePath,
      note: 'Screenshot saved. The loaded model is text-only and cannot view images itself — this is for the user to open.'
    };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

function bridgeState() {
  if (!browserView || browserView.webContents.isDestroyed()) {
    return { success: true, open: false };
  }
  return {
    success: true,
    open: true,
    url: browserView.webContents.getURL(),
    title: browserView.webContents.getTitle()
  };
}

function bridgeConfirmFileOperation(args) {
  return new Promise((resolve) => {
    if (!win || win.isDestroyed()) {
      resolve({ approved: false, error: 'No window available to request authorization.' });
      return;
    }
    const request = {
      id: Date.now().toString(36) + Math.random().toString(36).slice(2, 7),
      description: String((args && args.description) || 'File operation').slice(0, 500),
      operations: Array.isArray(args && args.operations) ? args.operations.slice(0, 200) : []
    };
    pendingFileConfirmations.set(request.id, (approved) => {
      resolve({ approved: !!approved });
    });
    safeSend('file-operation-request', request);
  });
}

function bridgeConfirmAppOperation(args) {
  return new Promise((resolve) => {
    if (!win || win.isDestroyed()) {
      resolve({ approved: false, error: 'No window available to request authorization.' });
      return;
    }
    const request = {
      id: Date.now().toString(36) + Math.random().toString(36).slice(2, 7),
      action: String((args && args.action) || 'install').slice(0, 20),
      name: String((args && args.name) || '').slice(0, 200),
      package_id: String((args && args.package_id) || '').slice(0, 200)
    };
    pendingAppConfirmations.set(request.id, (approved) => {
      resolve({ approved: !!approved });
    });
    safeSend('app-operation-request', request);
  });
}

function bridgeScheduleWorkflow(args) {
  try {
    const name = String((args && args.name) || '').trim();
    const prompt = String((args && args.prompt) || name).trim();
    if (!name) return { success: false, error: 'name is required' };
    if (!prompt) return { success: false, error: 'prompt is required' };

    let cronExpression = String((args && args.cron_expression) || '').trim();
    if (!cronExpression) {
      const time = String((args && args.time) || '').trim();
      const intervalMinutes = parseInt(args && args.interval_minutes, 10);
      if (time && /^\d{1,2}:\d{2}$/.test(time)) {
        const [hh, mm] = time.split(':').map(Number);
        if (hh >= 0 && hh <= 23 && mm >= 0 && mm <= 59) {
          cronExpression = `${mm} ${hh} * * *`;
        }
      } else if (!isNaN(intervalMinutes) && intervalMinutes > 0) {
        if (intervalMinutes % 60 === 0) {
          const hours = intervalMinutes / 60;
          if (hours >= 24) {
            return { success: false, error: 'interval_minutes is too large; pass a cron_expression instead.' };
          }
          cronExpression = `0 */${hours} * * *`;
        } else if (intervalMinutes < 60) {
          cronExpression = `*/${intervalMinutes} * * * *`;
        } else {
          return {
            success: false,
            error: 'interval_minutes must be under 60 or an exact multiple of 60; for other intervals pass a raw cron_expression.'
          };
        }
      }
    }
    if (!cronExpression || !workflowStorage.isValidCron(cronExpression)) {
      return {
        success: false,
        error: 'Could not determine a valid schedule. Provide exactly one of: time ("HH:MM" for a daily schedule), interval_minutes (repeating interval), or a raw cron_expression.'
      };
    }

    const jobs = workflowStorage.loadJobs();
    const newJob = {
      id: Date.now().toString(36) + Math.random().toString(36).slice(2, 7),
      name,
      cronExpression,
      workflow: { type: 'ai-prompt', prompt, outputAction: 'chat' },
      enabled: true
    };
    jobs.push(newJob);
    workflowStorage.saveJobs(jobs);
    workflowScheduler.scheduleJob(newJob, executeWorkflow);
    console.log('⏰ Workflow scheduled via AI tool:', newJob.name, `(${cronExpression})`);
    return { success: true, job: { id: newJob.id, name: newJob.name, cronExpression: newJob.cronExpression } };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

function bridgeUnscheduleWorkflow(args) {
  try {
    const match = String((args && (args.name || args.id)) || '').trim();
    if (!match) return { success: false, error: 'name or id is required' };
    let jobs = workflowStorage.loadJobs();
    const target = jobs.find((j) => j.id === match || j.name.toLowerCase() === match.toLowerCase());
    if (!target) return { success: false, error: `No workflow found matching "${match}"` };
    jobs = jobs.filter((j) => j.id !== target.id);
    workflowStorage.saveJobs(jobs);
    workflowScheduler.unscheduleJob(target.id);
    console.log('⏰ Workflow unscheduled via AI tool:', target.name);
    return { success: true, removed: target.name };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

function bridgeListWorkflows() {
  try {
    const jobs = workflowStorage.loadJobs();
    return {
      success: true,
      count: jobs.length,
      workflows: jobs.map((j) => ({
        id: j.id,
        name: j.name,
        cronExpression: j.cronExpression,
        enabled: j.enabled,
        type: (j.workflow && j.workflow.type) || 'unknown'
      }))
    };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

function bridgeListTools() {
  try {
    const tools = toolsStorage.loadTools();
    return {
      success: true,
      count: tools.length,
      tools: tools.map((t) => ({
        name: t.name,
        description: t.description || '',
        parameters: t.parameters && typeof t.parameters === 'object'
          ? t.parameters
          : { type: 'object', properties: {}, required: [] }
      }))
    };
  } catch (err) {
    return { success: false, error: err.message, tools: [] };
  }
}

async function bridgeRunTool(args) {
  try {
    const name = String((args && args.name) || '').trim();
    if (!name) return { success: false, error: 'name is required' };
    const tools = toolsStorage.loadTools();
    const tool = tools.find((t) => t.name === name || t.id === name);
    if (!tool) return { success: false, error: `Tool "${name}" not found.` };
    return await toolRunner.executeTool(tool, args.params || {});
  } catch (err) {
    return { success: false, error: err.message };
  }
}

async function bridgeRunOrchestrator(args) {
  try {
    if (typeof toolRunner.executeOrchestrator !== 'function') {
      return { success: false, error: 'Orchestrator not available — update toolRunner.js' };
    }
    return await toolRunner.executeOrchestrator(String((args && args.script) || ''));
  } catch (err) {
    return { success: false, error: err.message };
  }
}

// ==============================================================
// BROWSER CONTROL BRIDGE SERVER
// ==============================================================
function startBridgeServer() {
  bridgeServer = http.createServer((req, res) => {
    if (req.headers['x-bridge-token'] !== bridgeToken) {
      res.writeHead(403, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ success: false, error: 'Forbidden' }));
      return;
    }
    let body = '';
    req.on('data', (chunk) => { body += chunk; });
    req.on('end', async () => {
      let args = {};
      try { args = body ? JSON.parse(body) : {}; } catch (e) { /* treat as no args */ }

      let result;
      try {
        if (req.method === 'POST' && req.url === '/browser/navigate') {
          result = await bridgeNavigate(args);
        } else if (req.method === 'POST' && req.url === '/browser/click') {
          result = await bridgeClick(args);
        } else if (req.method === 'POST' && req.url === '/browser/fill') {
          result = await bridgeFill(args);
        } else if (req.method === 'POST' && req.url === '/browser/read') {
          result = await bridgeRead(args);
        } else if (req.method === 'POST' && req.url === '/browser/screenshot') {
          result = await bridgeScreenshot();
        } else if (req.method === 'GET' && req.url === '/browser/state') {
          result = await bridgeState();
        } else if (req.method === 'POST' && req.url === '/files/confirm') {
          result = await bridgeConfirmFileOperation(args);
        } else if (req.method === 'POST' && req.url === '/system/confirm') {
          result = await bridgeConfirmAppOperation(args);
        } else if (req.method === 'POST' && req.url === '/workflows/schedule') {
          result = bridgeScheduleWorkflow(args);
        } else if (req.method === 'POST' && req.url === '/workflows/unschedule') {
          result = bridgeUnscheduleWorkflow(args);
        } else if (req.method === 'GET' && req.url === '/workflows/list') {
          result = bridgeListWorkflows();
        } else if (req.method === 'GET' && req.url === '/tools/list') {
          result = bridgeListTools();
        } else if (req.method === 'POST' && req.url === '/tools/run') {
          result = await bridgeRunTool(args);
        } else if (req.method === 'POST' && req.url === '/orchestrator/run') {
          result = await bridgeRunOrchestrator(args);
        } else if (req.method === 'POST' && req.url === '/sky/open') {
          result = await bridgeOpenSky(args);
        } else {
          res.writeHead(404, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ success: false, error: 'Not found' }));
          return;
        }
      } catch (err) {
        result = { success: false, error: err.message };
      }

      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(result));
    });
  });
  bridgeServer.on('error', (err) => {
    console.error('❌ Browser control bridge failed to start:', err.message);
  });
  bridgeServer.listen(bridgePort, '127.0.0.1', () => {
    console.log(`🌉 Browser control bridge listening on 127.0.0.1:${bridgePort}`);
  });
}

ipcMain.on('close-browser', () => closeBrowserView());

ipcMain.on('browser-go-back', () => {
  if (browserView && !browserView.webContents.isDestroyed() && browserIndex > 0) {
    browserIndex--;
    browserView.webContents.goBack();
  }
});

ipcMain.on('browser-go-forward', () => {
  if (browserView && !browserView.webContents.isDestroyed() && browserIndex < browserHistory.length - 1) {
    browserIndex++;
    browserView.webContents.goForward();
  }
});

ipcMain.on('browser-refresh', () => {
  if (browserView && !browserView.webContents.isDestroyed()) browserView.webContents.reload();
});

ipcMain.on('browser-load-url', (event, url) => {
  if (browserView && !browserView.webContents.isDestroyed()) {
    let finalUrl = url.trim();
    if (!finalUrl.startsWith('http://') && !finalUrl.startsWith('https://')) {
      if (finalUrl.includes(' ') || !finalUrl.includes('.')) {
        finalUrl = 'https://www.google.com/search?q=' + encodeURIComponent(finalUrl);
      } else {
        finalUrl = 'https://' + finalUrl;
      }
    }
    browserView.webContents.loadURL(finalUrl);
    browserHistory.push(finalUrl);
    browserIndex = browserHistory.length - 1;
  }
});

ipcMain.on('browser-search', (event, query) => {
  if (browserView && !browserView.webContents.isDestroyed()) {
    const searchUrl = 'https://www.google.com/search?q=' + encodeURIComponent(query);
    browserView.webContents.loadURL(searchUrl);
    browserHistory.push(searchUrl);
    browserIndex = browserHistory.length - 1;
  }
});

console.log('✅ J.A.R.V.I.S. Main process ready');