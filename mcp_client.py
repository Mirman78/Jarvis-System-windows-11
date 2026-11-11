"""Minimal MCP (Model Context Protocol) client for JARVIS.

Speaks JSON-RPC 2.0 over stdio to an MCP server subprocess. Implements the
core subset JARVIS needs:
  - initialize handshake
  - tools/list
  - tools/call

Deliberately does NOT implement resources/prompts/sampling/roots — that's
the long tail of the spec, and tools alone cover 90% of the MCP servers in
the wild (filesystem, git, sqlite, memory, fetch, time, github, slack, …).

Config format matches Claude Desktop's `mcpServers` schema exactly, so any
config the user finds in the ecosystem can be copy-pasted straight in.
"""

import json
import os
import platform
import subprocess
import threading
import logging

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2024-11-05"
DEFAULT_REQUEST_TIMEOUT = 30
INIT_TIMEOUT = 20


class MCPError(Exception):
    """Raised when an MCP request fails."""


class MCPClient:
    """One MCP server subprocess + a JSON-RPC channel to it."""

    def __init__(self, name, command, args=None, env=None, cwd=None):
        self.name = name
        self.command = command
        self.args = list(args or [])
        self.extra_env = dict(env or {})
        self.cwd = cwd
        self.proc = None
        self.tools = []
        self.server_info = {}
        self._lock = threading.Lock()
        self._pending = {}          # request_id -> {event, response, error}
        self._reader_thread = None
        self._reader_alive = False
        self._next_id = 0

    def start(self):
        """Spawn the server and do the initialize handshake. Raises on failure."""
        env = os.environ.copy()
        env.update(self.extra_env)

        # Windows npm/npx/yarn shims are .cmd files, so `spawn("npx", ...)`
        # fails with FileNotFoundError. Wrap in cmd /c so the shim resolves.
        cmd_list = [self.command] + self.args
        if platform.system() == 'Windows' and not self.command.lower().endswith('.exe'):
            cmd_list = ['cmd', '/c'] + cmd_list

        try:
            self.proc = subprocess.Popen(
                cmd_list,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=self.cwd,
                text=False,
                bufsize=0,
            )
        except FileNotFoundError:
            raise MCPError(f"Command not found: {self.command} — is it on PATH?")

        self._reader_alive = True
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name=f"mcp-reader-{self.name}"
        )
        self._reader_thread.start()
        threading.Thread(
            target=self._stderr_loop, daemon=True, name=f"mcp-stderr-{self.name}"
        ).start()

        init_result = self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "jarvis", "version": "1.0"}
        }, timeout=INIT_TIMEOUT)
        self.server_info = init_result.get("serverInfo", {})

        self._notify("notifications/initialized", {})
        self._refresh_tools()

        logger.info(
            f"✅ MCP server '{self.name}' started: "
            f"{self.server_info.get('name', '?')} v{self.server_info.get('version', '?')}, "
            f"{len(self.tools)} tool(s)"
        )
        return True

    def _refresh_tools(self):
        result = self._request("tools/list", {})
        raw_tools = result.get("tools", [])
        self.tools = []
        for t in raw_tools:
            name = t.get("name", "")
            if not name:
                continue
            self.tools.append({
                "name": f"mcp__{self.name}__{name}",
                "description": (t.get("description") or "")[:500],
                "parameters": t.get("inputSchema") or {"type": "object", "properties": {}},
            })

    def call_tool(self, real_tool_name, arguments):
        """Call a tool by its ORIGINAL (non-namespaced) name."""
        result = self._request("tools/call", {
            "name": real_tool_name,
            "arguments": arguments or {}
        }, timeout=60)

        content = result.get("content", [])
        texts = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text":
                    texts.append(c.get("text", ""))
                elif c.get("type") == "image":
                    texts.append(f"[image: {c.get('mimeType', 'unknown')}]")
                elif c.get("type") == "resource":
                    texts.append(f"[resource: {c.get('uri', '?')}]")
        combined = "\n".join(texts) if texts else "(no content)"

        return {
            "success": not result.get("isError", False),
            "content": combined[:8000],
            "content_items": len(content)
        }

    def _reader_loop(self):
        try:
            for line in iter(self.proc.stdout.readline, b''):
                if not line:
                    break
                try:
                    msg = json.loads(line.decode('utf-8', errors='replace'))
                except Exception as e:
                    logger.warning(f"MCP[{self.name}] malformed message: {e}")
                    continue
                self._handle_message(msg)
        except Exception as e:
            logger.error(f"MCP[{self.name}] reader crashed: {e}")
        finally:
            self._reader_alive = False
            with self._lock:
                for rid, pending in list(self._pending.items()):
                    pending['error'] = "Server connection closed"
                    pending['event'].set()
                self._pending.clear()

    def _stderr_loop(self):
        try:
            for line in iter(self.proc.stderr.readline, b''):
                if not line:
                    break
                text = line.decode('utf-8', errors='replace').rstrip()
                if text:
                    logger.info(f"MCP[{self.name}] stderr: {text}")
        except Exception:
            pass

    def _handle_message(self, msg):
        if 'id' in msg and 'method' not in msg:
            with self._lock:
                pending = self._pending.pop(msg['id'], None)
            if pending:
                if 'error' in msg:
                    pending['error'] = msg['error'].get('message', 'Unknown MCP error')
                else:
                    pending['response'] = msg.get('result', {})
                pending['event'].set()
            return

        method = msg.get('method', '')
        if method.startswith('notifications/'):
            return

        if 'id' in msg:
            self._send({
                "jsonrpc": "2.0",
                "id": msg['id'],
                "error": {"code": -32601, "message": f"Unsupported method: {method}"}
            })

    def _send(self, msg):
        if not self.proc or self.proc.poll() is not None:
            raise MCPError(f"MCP server '{self.name}' is not running")
        data = (json.dumps(msg) + "\n").encode('utf-8')
        try:
            self.proc.stdin.write(data)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise MCPError(f"MCP[{self.name}] write failed: {e}")

    def _notify(self, method, params):
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _request(self, method, params, timeout=DEFAULT_REQUEST_TIMEOUT):
        with self._lock:
            self._next_id += 1
            rid = self._next_id
            pending = {'event': threading.Event(), 'response': None, 'error': None}
            self._pending[rid] = pending

        try:
            self._send({
                "jsonrpc": "2.0",
                "id": rid,
                "method": method,
                "params": params or {}
            })
        except Exception:
            with self._lock:
                self._pending.pop(rid, None)
            raise

        if not pending['event'].wait(timeout):
            with self._lock:
                self._pending.pop(rid, None)
            raise MCPError(f"MCP[{self.name}] '{method}' timed out after {timeout}s")

        if pending['error']:
            raise MCPError(f"MCP[{self.name}] error in '{method}': {pending['error']}")
        return pending['response'] or {}

    def stop(self):
        self._reader_alive = False
        if not self.proc:
            return
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        except Exception:
            pass
        self.proc = None
        logger.info(f"🛑 MCP server '{self.name}' stopped")


class MCPRegistry:
    """Owns every MCP client and dispatches namespaced tool calls."""

    def __init__(self):
        self.clients = {}
        self.tool_map = {}   # namespaced_name -> (client, real_name)
        self._lock = threading.Lock()

    def load_from_config(self, config_path):
        if not config_path or not os.path.exists(config_path):
            logger.info("   (No MCP servers configured — mcp_servers.json not found)")
            return 0

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read MCP config {config_path}: {e}")
            return 0

        servers = cfg.get('mcpServers', {})
        if not isinstance(servers, dict):
            logger.error("MCP config 'mcpServers' must be an object")
            return 0

        count = 0
        for name, spec in servers.items():
            if not isinstance(spec, dict):
                continue
            command = spec.get('command')
            if not command:
                logger.warning(f"MCP server '{name}' has no 'command' — skipping")
                continue
            try:
                client = MCPClient(
                    name=name,
                    command=command,
                    args=spec.get('args', []),
                    env=spec.get('env', {}),
                    cwd=spec.get('cwd'),
                )
                client.start()
                with self._lock:
                    self.clients[name] = client
                    for t in client.tools:
                        real = t['name'].split('__', 2)[2]
                        self.tool_map[t['name']] = (client, real)
                count += 1
            except Exception as e:
                logger.error(f"❌ MCP server '{name}' failed to start: {e}")
        return count

    def get_all_tools(self):
        with self._lock:
            tools = []
            for c in self.clients.values():
                tools.extend(c.tools)
            return tools

    def call_tool(self, namespaced_name, arguments):
        with self._lock:
            entry = self.tool_map.get(namespaced_name)
        if not entry:
            return {"error": f"Unknown MCP tool: {namespaced_name}"}
        client, real = entry
        try:
            return client.call_tool(real, arguments)
        except Exception as e:
            return {"error": str(e)}

    def stop_all(self):
        with self._lock:
            clients = list(self.clients.values())
        for c in clients:
            try:
                c.stop()
            except Exception:
                pass
        with self._lock:
            self.clients.clear()
            self.tool_map.clear()

    def status(self):
        with self._lock:
            return [
                {
                    "name": name,
                    "server": c.server_info.get('name', '?'),
                    "version": c.server_info.get('version', '?'),
                    "tools": len(c.tools),
                    "running": c.proc is not None and c.proc.poll() is None,
                }
                for name, c in self.clients.items()
            ]