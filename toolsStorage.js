const fs = require('fs');
const path = require('path');
const { app } = require('electron');

// Use Electron's per-user writable data directory rather than a path next
// to __dirname: in a packaged build __dirname resolves inside the read-only
// asar archive, so writes there would throw. (Note: workflows.json and
// chat_data/ in this project currently still write next to __dirname and
// would hit the same problem once packaged — worth moving those too.)
const TOOLS_FILE = path.join(app.getPath('userData'), 'tools.json');

function loadTools() {
  if (!fs.existsSync(TOOLS_FILE)) return [];
  try {
    const data = fs.readFileSync(TOOLS_FILE, 'utf-8');
    const parsed = JSON.parse(data);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((t) => t && typeof t === 'object' && t.id && t.name && typeof t.handlerCode === 'string');
  } catch (err) {
    console.error('Failed to load tools.json:', err.message);
    const backup = TOOLS_FILE + '.backup';
    if (fs.existsSync(backup)) {
      try {
        const backupData = fs.readFileSync(backup, 'utf-8');
        const parsed = JSON.parse(backupData);
        if (Array.isArray(parsed)) {
          console.log('Restored tools from backup.');
          return parsed;
        }
      } catch (backupErr) {
        console.error('Tools backup also corrupt.');
      }
    }
    return [];
  }
}

function saveTools(tools) {
  const tempPath = TOOLS_FILE + '.tmp';
  const backupPath = TOOLS_FILE + '.backup';
  try {
    fs.writeFileSync(tempPath, JSON.stringify(tools, null, 2), 'utf-8');
    if (fs.existsSync(TOOLS_FILE)) {
      fs.copyFileSync(TOOLS_FILE, backupPath);
    }
    fs.renameSync(tempPath, TOOLS_FILE);
  } catch (err) {
    console.error('Failed to save tools:', err.message);
    throw err;
  }
}

module.exports = { loadTools, saveTools, TOOLS_FILE };
