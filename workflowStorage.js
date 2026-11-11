const fs = require('fs');
const path = require('path');
const cron = require('node-cron');
const { app } = require('electron');

const JOBS_FILE = path.join(app.getPath('userData'), 'workflows.json');
const LEGACY_JOBS_FILE = path.join(__dirname, 'workflows.json');

// One-time migration from the pre-userData location. Same rationale as
// memoryManager.js: __dirname is read-only inside a packaged asar.
function migrateLegacyJobsIfNeeded() {
  try {
    if (!fs.existsSync(LEGACY_JOBS_FILE)) return;
    if (path.resolve(LEGACY_JOBS_FILE) === path.resolve(JOBS_FILE)) return;
    if (fs.existsSync(JOBS_FILE)) return;
    fs.mkdirSync(path.dirname(JOBS_FILE), { recursive: true });
    fs.copyFileSync(LEGACY_JOBS_FILE, JOBS_FILE);
    console.log(`📦 Migrated legacy workflows.json → ${JOBS_FILE}`);
  } catch (e) {
    console.warn(`⚠️ Could not migrate legacy workflows.json: ${e.message}`);
  }
}

function loadJobs() {
  migrateLegacyJobsIfNeeded();
  if (!fs.existsSync(JOBS_FILE)) return [];
  try {
    const data = fs.readFileSync(JOBS_FILE, 'utf-8');
    const parsed = JSON.parse(data);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((job) =>
      job && typeof job === 'object' && job.id && job.cronExpression && isValidCron(job.cronExpression)
    );
  } catch (err) {
    console.error('Failed to load workflows.json:', err.message);
    const backup = JOBS_FILE + '.backup';
    if (fs.existsSync(backup)) {
      try {
        const backupData = fs.readFileSync(backup, 'utf-8');
        const parsed = JSON.parse(backupData);
        if (Array.isArray(parsed)) {
          console.log('Restored from backup.');
          return parsed;
        }
      } catch (backupErr) {
        console.error('Backup also corrupt.');
      }
    }
    return [];
  }
}

function saveJobs(jobs) {
  const tempPath = JOBS_FILE + '.tmp';
  const backupPath = JOBS_FILE + '.backup';
  try {
    fs.mkdirSync(path.dirname(JOBS_FILE), { recursive: true });
    fs.writeFileSync(tempPath, JSON.stringify(jobs, null, 2), 'utf-8');
    if (fs.existsSync(JOBS_FILE)) {
      fs.copyFileSync(JOBS_FILE, backupPath);
    }
    fs.renameSync(tempPath, JOBS_FILE);
  } catch (err) {
    console.error('Failed to save workflows:', err.message);
    throw err;
  }
}

function isValidCron(expression) {
  return typeof expression === 'string' && cron.validate(expression);
}

module.exports = { loadJobs, saveJobs, isValidCron, JOBS_FILE };