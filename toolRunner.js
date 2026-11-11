const vm = require('vm');
const http = require('http');

// SECURITY NOTE: Node's `vm` module is explicitly documented as NOT a
// security boundary — determined malicious code can still escape it. The
// real protection here is the human-in-the-loop confirmation step in
// index.html: nothing in this file runs until a person has read the
// handlerCode and clicked "Authorize". This module just keeps well-behaved
// (or buggy) tool code from casually touching the filesystem, environment,
// or other Node internals, and stops runaway loops from freezing the app.

const SYNC_TIMEOUT_MS = 5000;
const OVERALL_TIMEOUT_MS = 15000;

/**
 * Execute a saved tool's handlerCode with the given params.
 */
function executeTool(tool, params) {
  return new Promise((resolve) => {
    let settled = false;
    const settle = (value) => {
      if (!settled) {
        settled = true;
        resolve(value);
      }
    };

    const overallTimer = setTimeout(() => {
      settle({ success: false, error: `Tool timed out after ${OVERALL_TIMEOUT_MS}ms` });
    }, OVERALL_TIMEOUT_MS);

    const logs = [];
    try {
      const sandbox = {
        params: params && typeof params === 'object' ? params : {},
        console: { log: (...args) => logs.push(args.map((a) => (typeof a === 'string' ? a : JSON.stringify(a))).join(' ')) },
        fetch: typeof fetch === 'function' ? fetch : undefined,
        Promise,
        setTimeout,
        JSON,
        Math,
        Date,
        URL,
        URLSearchParams,
        TextEncoder,
        TextDecoder,
        encodeURIComponent,
        decodeURIComponent
      };
      const context = vm.createContext(sandbox);
      const wrapped = `(async () => {\n${tool.handlerCode}\n})()`;
      const script = new vm.Script(wrapped, { filename: `tool-${tool.name}.js`, timeout: SYNC_TIMEOUT_MS });
      const returned = script.runInContext(context, { timeout: SYNC_TIMEOUT_MS });

      Promise.resolve(returned)
        .then((value) => {
          clearTimeout(overallTimer);
          settle({ success: true, result: value, logs });
        })
        .catch((err) => {
          clearTimeout(overallTimer);
          settle({ success: false, error: err && err.message ? err.message : String(err), logs });
        });
    } catch (err) {
      clearTimeout(overallTimer);
      settle({ success: false, error: err && err.message ? err.message : String(err), logs });
    }
  });
}

// ==============================================================
// ORCHESTRATOR — model-authored JS that composes multiple tool calls
// ==============================================================
//
// An orchestrator script runs inside a vm sandbox with one privileged
// binding: `jarvis.callTool(name, args)`. That function does NOT call
// tools directly — it hits the Python server's `/tools/exec` endpoint,
// which is what actually has the tool implementations. Two layers of
// safety on top of the vm:
//
//   1. The allowlist below restricts which tools can be called from
//      inside a script — only read-only ones. This is enforced on the
//      Node side, BEFORE any HTTP request leaves the process, so even
//      a malicious script can't ask the Python server to install apps
//      or run shell commands.
//
//   2. `/tools/exec` itself is authenticated with X-Bridge-Token, so
//      nothing else on the machine can reach it without the token.

let orchestratorConfig = null;

function initOrchestrator(config) {
  if (!config || !config.port || !config.token) {
    throw new Error('initOrchestrator requires { port, token }');
  }
  orchestratorConfig = { port: config.port, token: config.token };
}

// The full set of tools callable from inside an orchestrator script.
// Deliberately read-only: nothing here mutates the filesystem, installs
// software, runs shell commands, or schedules workflows. Interactive
// tools (browser_navigate/click/fill) are also excluded since they'd
// need their own confirmation round-trip and complicate the sandbox.
const ORCHESTRATOR_ALLOWED_TOOLS = new Set([
  'web_search',
  'search_knowledge_base',
  'recall_memories',
  'find_files',
  'list_directory',
  'read_file_content',
  'list_open_windows',
  'read_window',
  'get_system_info',
  'list_installed_applications',
  'search_applications',
  'list_workflows',
]);

const ORCH_SYNC_TIMEOUT_MS = 5000;
const ORCH_OVERALL_TIMEOUT_MS = 60000;

function callToolFromOrchestrator(name, args) {
  return new Promise((resolve, reject) => {
    if (!orchestratorConfig) {
      reject(new Error('Orchestrator not initialized'));
      return;
    }
    const postData = JSON.stringify({ name, arguments: args || {} });
    const req = http.request({
      hostname: '127.0.0.1',
      port: orchestratorConfig.port,
      path: '/tools/exec',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData),
        'X-Bridge-Token': orchestratorConfig.token,
      },
      timeout: 30000,
    }, (res) => {
      let body = '';
      res.on('data', (chunk) => { body += chunk; });
      res.on('end', () => {
        try {
          resolve(JSON.parse(body));
        } catch (e) {
          reject(new Error(`Invalid response from tool server: ${e.message}`));
        }
      });
    });
    req.on('error', reject);
    req.on('timeout', () => req.destroy(new Error('Tool call timed out (30s)')));
    req.write(postData);
    req.end();
  });
}

function executeOrchestrator(script) {
  return new Promise((resolve) => {
    let settled = false;
    const settle = (value) => {
      if (!settled) {
        settled = true;
        resolve(value);
      }
    };

    if (typeof script !== 'string' || !script.trim()) {
      settle({ success: false, error: 'script is required' });
      return;
    }
    if (!orchestratorConfig) {
      settle({ success: false, error: 'Orchestrator not initialized — check main.js startup' });
      return;
    }

    const logs = [];
    const overallTimer = setTimeout(() => {
      settle({
        success: false,
        error: `Orchestrator timed out after ${ORCH_OVERALL_TIMEOUT_MS}ms`,
        logs,
      });
    }, ORCH_OVERALL_TIMEOUT_MS);

    try {
      const sandbox = {
        jarvis: {
          callTool: (name, args) => {
            const strName = String(name || '');
            if (!ORCHESTRATOR_ALLOWED_TOOLS.has(strName)) {
              return Promise.reject(new Error(
                `Tool '${strName}' is not allowed inside the orchestrator ` +
                `(only read-only tools are permitted).`
              ));
            }
            return callToolFromOrchestrator(strName, args);
          },
        },
        console: {
          log: (...a) => logs.push(a.map((x) => (typeof x === 'string' ? x : JSON.stringify(x))).join(' ')),
          warn: (...a) => logs.push('WARN: ' + a.map((x) => (typeof x === 'string' ? x : JSON.stringify(x))).join(' ')),
          error: (...a) => logs.push('ERROR: ' + a.map((x) => (typeof x === 'string' ? x : JSON.stringify(x))).join(' ')),
        },
        Promise,
        setTimeout,
        JSON,
        Math,
        Date,
        Array,
        Object,
        String,
        Number,
        Boolean,
        encodeURIComponent,
        decodeURIComponent,
      };
      const context = vm.createContext(sandbox);
      const wrapped = `(async () => {\n${script}\n})()`;
      const compiled = new vm.Script(wrapped, {
        filename: 'orchestrator.js',
        timeout: ORCH_SYNC_TIMEOUT_MS,
      });
      const returned = compiled.runInContext(context, { timeout: ORCH_SYNC_TIMEOUT_MS });

      Promise.resolve(returned)
        .then((value) => {
          clearTimeout(overallTimer);
          settle({ success: true, result: value, logs });
        })
        .catch((err) => {
          clearTimeout(overallTimer);
          settle({ success: false, error: err && err.message ? err.message : String(err), logs });
        });
    } catch (err) {
      clearTimeout(overallTimer);
      settle({ success: false, error: err && err.message ? err.message : String(err), logs });
    }
  });
}

module.exports = { executeTool, initOrchestrator, executeOrchestrator };