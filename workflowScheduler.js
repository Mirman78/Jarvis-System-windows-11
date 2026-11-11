const cron = require('node-cron');

// Keeps the live node-cron ScheduledTask objects, keyed by job id, so that
// unscheduling can actually stop them (previously nothing was ever
// scheduled at all — jobs just sat in workflows.json doing nothing).
const activeTasks = new Map();

/**
 * Start (or restart) a cron task for a job.
 * @param {object} job - { id, name, cronExpression, enabled }
 * @param {(job: object) => void} onTick - called every time the cron fires
 */
function scheduleJob(job, onTick) {
  unscheduleJob(job.id);
  if (!job.enabled) return;
  if (!cron.validate(job.cronExpression)) {
    console.warn(`Skipping schedule for "${job.name}": invalid cron expression`);
    return;
  }
  const task = cron.schedule(job.cronExpression, () => {
    try {
      onTick(job);
    } catch (err) {
      console.error(`Error running scheduled job "${job.name}":`, err);
    }
  });
  activeTasks.set(job.id, task);
  console.log(`⏰ Cron task active for "${job.name}" (${job.cronExpression})`);
}

function unscheduleJob(id) {
  const task = activeTasks.get(id);
  if (task) {
    task.stop();
    activeTasks.delete(id);
  }
}

function unscheduleAll() {
  for (const id of activeTasks.keys()) unscheduleJob(id);
}

/**
 * Load-time bootstrap: schedule every enabled job from storage.
 * @param {object[]} jobs
 * @param {(job: object) => void} onTick
 */
function initAll(jobs, onTick) {
  unscheduleAll();
  for (const job of jobs) {
    scheduleJob(job, onTick);
  }
}

module.exports = { scheduleJob, unscheduleJob, unscheduleAll, initAll };
