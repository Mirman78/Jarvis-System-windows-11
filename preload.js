const { contextBridge, ipcRenderer } = require('electron');

function on(channel, callback) {
  const listener = (_event, data) => callback(data);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
}

contextBridge.exposeInMainWorld('electronAPI', {
  minimize: () => ipcRenderer.send('minimize'),
  close: () => ipcRenderer.send('close'),

  openVideo: (url) => ipcRenderer.send('open-video', url),
  closeVideo: () => ipcRenderer.send('close-video'),

  openMap: (mapData) => ipcRenderer.send('open-map', mapData),
  closeMap: () => ipcRenderer.send('close-map'),

  // NEW: Sky View
  openSky: (payload) => ipcRenderer.send('open-sky', payload || {}),
  closeSky: () => ipcRenderer.send('close-sky'),
  onSkyOpened: (cb) => on('sky-opened', cb),
  onSkyClosed: (cb) => on('sky-closed', cb),
  onSkyError:  (cb) => on('sky-error', cb),

  openBrowser: (url) => ipcRenderer.send('open-browser', url),
  closeBrowser: () => ipcRenderer.send('close-browser'),
  browserGoBack: () => ipcRenderer.send('browser-go-back'),
  browserGoForward: () => ipcRenderer.send('browser-go-forward'),
  browserRefresh: () => ipcRenderer.send('browser-refresh'),
  browserLoadURL: (url) => ipcRenderer.send('browser-load-url', url),
  browserSearch: (query) => ipcRenderer.send('browser-search', query),

  getHistory: (sessionId) => ipcRenderer.invoke('get-history', sessionId),
  addMessage: (sessionId, role, content) =>
    ipcRenderer.invoke('add-message', { sessionId, role, content }),

  listSessions: () => ipcRenderer.invoke('list-sessions'),
  createSession: (title) => ipcRenderer.invoke('create-session', title),
  deleteSession: (id) => ipcRenderer.invoke('delete-session', id),
  renameSession: (id, title) => ipcRenderer.invoke('rename-session', { id, title }),

  aiServerStatus: () => ipcRenderer.invoke('ai-server-status'),
  aiListModels: () => ipcRenderer.invoke('ai-list-models'),

  aiLoadModel: (modelName, options) =>
    ipcRenderer.invoke('ai-load-model', { model: modelName, ...(options || {}) }),
  aiUnloadModel: () => ipcRenderer.invoke('ai-unload-model'),

  aiGenerate: (prompt, options) =>
    ipcRenderer.invoke('ai-generate', { prompt, ...(options || {}) }),
  aiAbort: () => ipcRenderer.invoke('ai-abort'),

  selectModelFile: () => ipcRenderer.invoke('select-model-file'),
  reconnectAIServer: () => ipcRenderer.invoke('reconnect-ai-server'),

  aiListMemories: (query) => ipcRenderer.invoke('ai-list-memories', query),
  aiAddMemory: (fact, tags) => {
    if (typeof fact === 'object' && fact !== null && !Array.isArray(fact)) {
      return ipcRenderer.invoke('ai-add-memory', { ...fact });
    }
    return ipcRenderer.invoke('ai-add-memory', { fact, tags });
  },
  aiDeleteMemory: (idOrFact) => ipcRenderer.invoke('ai-delete-memory', idOrFact),

  aiKbList: () => ipcRenderer.invoke('ai-kb-list'),
  aiKbSearch: (query) => ipcRenderer.invoke('ai-kb-search', query),
  aiKbAdd: (filePath) => ipcRenderer.invoke('ai-kb-add', filePath),
  aiKbRemove: (match) => ipcRenderer.invoke('ai-kb-remove', match),

  aiBrowserNavigate: (url) =>
    ipcRenderer.invoke('ai-browser-navigate',
      typeof url === 'string' ? { url } : (url || {})),
  aiBrowserClick: (args) => ipcRenderer.invoke('ai-browser-click', args || {}),
  aiBrowserFill: (args) => ipcRenderer.invoke('ai-browser-fill', args || {}),
  aiBrowserRead: (args) => ipcRenderer.invoke('ai-browser-read', args || {}),
  aiBrowserScreenshot: () => ipcRenderer.invoke('ai-browser-screenshot'),

  aiFilesFind: (args) => ipcRenderer.invoke('ai-files-find', args || {}),
  aiFilesList: (directory) => ipcRenderer.invoke('ai-files-list', directory),
  aiFilesRead: (filePath) => ipcRenderer.invoke('ai-files-read', filePath),

  aiAppsSearch: (query) => ipcRenderer.invoke('ai-apps-search', query),
  aiAppsList: (query) => ipcRenderer.invoke('ai-apps-list', query),

  aiWebSearch: (query, maxResults) => ipcRenderer.invoke('ai-web-search', query, maxResults),

  readFile: (filePath) => ipcRenderer.invoke('read-file', filePath),
  writeFile: (filePath, content) =>
    ipcRenderer.invoke('write-file', { path: filePath, content }),
  listDirectory: (dirPath) => ipcRenderer.invoke('list-directory', dirPath),
  selectFile: (options) => ipcRenderer.invoke('select-file', options),
  selfCorrect: (payload) => ipcRenderer.invoke('self-correct', payload),

  scheduleWorkflow: (jobData) => ipcRenderer.invoke('schedule-workflow', jobData),
  unscheduleWorkflow: (id) => ipcRenderer.invoke('unschedule-workflow', id),
  listWorkflows: () => ipcRenderer.invoke('list-workflows'),
  runWorkflowNow: (id) => ipcRenderer.invoke('run-workflow-now', id),

  listTools: () => ipcRenderer.invoke('list-tools'),
  proposeTool: (toolDef) => ipcRenderer.invoke('propose-tool', toolDef),
  runTool: (name, params) => ipcRenderer.invoke('run-tool', { name, params }),
  deleteTool: (idOrName) => ipcRenderer.invoke('delete-tool', idOrName),

  getResourceStats: () => ipcRenderer.invoke('get-resource-stats'),

  onModelLoadStatus:     (cb) => on('ai-model-load-status', cb),
  onServerError:         (cb) => on('ai-server-error', cb),
  onServerReady:         (cb) => on('ai-server-ready', cb),
  onServerStatusUpdate:  (cb) => on('server-status-update', cb),
  onWorkflowResult:      (cb) => on('workflow-result', cb),
  onToolCreationRequest: (cb) => on('tool-creation-request', cb),
  onFileOperationRequest:(cb) => on('file-operation-request', cb),
  onAppOperationRequest: (cb) => on('app-operation-request', cb),

  onBrowserOpened:    (cb) => on('browser-opened', cb),
  onBrowserClosed:    (cb) => on('browser-closed', cb),
  onBrowserUrlChanged:(cb) => on('browser-url-changed', cb),

  toolConfirmationResponse: (approved) =>
    ipcRenderer.send('tool-confirmation-response', approved),
  fileOperationResponse: (approved) =>
    ipcRenderer.send('file-operation-response', approved),
  appOperationResponse: (approved) =>
    ipcRenderer.send('app-operation-response', approved)
});