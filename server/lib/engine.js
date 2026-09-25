'use strict';
// Engine process supervisor: spawns kiln-engine.exe (or the mock), keeps it alive,
// speaks JSON lines over stdin/stdout and re-emits parsed events.

const { spawn } = require('child_process');
const EventEmitter = require('events');
const readline = require('readline');
const fs = require('fs');
const path = require('path');

class Engine extends EventEmitter {
  constructor({ exe, args, mockScript, cwd }) {
    super();
    this.exe = exe;
    this.args = args || [];
    this.mockScript = mockScript;
    this.cwd = cwd;
    this.proc = null;
    this.state = 'stopped'; // stopped | starting | ready | down
    this.mock = false;
    this.restarts = 0;
    this.lastStart = 0;
    this.loading = null;
    this.stopping = false;
    this.pid = null;
  }

  start() {
    this.stopping = false;
    const haveExe = this.exe && fs.existsSync(this.exe);
    this.mock = !haveExe;
    let cmd, args;
    if (haveExe) { cmd = this.exe; args = this.args; }
    else { cmd = process.execPath; args = [this.mockScript]; }
    this.lastStart = Date.now();
    this.setState('starting');
    let proc;
    try {
      proc = spawn(cmd, args, { cwd: this.cwd || (haveExe ? path.dirname(this.exe) : undefined), stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true });
    } catch (e) {
      this.emit('log', `engine spawn failed: ${e.message}`);
      this.scheduleRestart();
      return;
    }
    this.proc = proc;
    this.pid = proc.pid;
    this.emit('log', `engine started: ${this.mock ? 'MOCK ' : ''}${cmd} ${args.join(' ')} (pid ${proc.pid})`);
    proc.stdout.setEncoding('utf8');
    proc.stderr.setEncoding('utf8');
    readline.createInterface({ input: proc.stdout, crlfDelay: Infinity }).on('line', (line) => this.onLine(line));
    readline.createInterface({ input: proc.stderr, crlfDelay: Infinity }).on('line', (line) => {
      if (line.trim()) this.emit('log', '[engine stderr] ' + line);
    });
    proc.stdin.on('error', (e) => this.emit('log', 'engine stdin error: ' + e.message));
    proc.on('error', (e) => {
      this.emit('log', 'engine process error: ' + e.message);
    });
    proc.on('exit', (code, sig) => {
      if (this.proc !== proc) return;
      this.proc = null;
      this.pid = null;
      const stopping = this.stopping;
      this.emit('log', `engine exited (code ${code}${sig ? ', signal ' + sig : ''})`);
      if (!stopping) this.scheduleRestart();
      else this.setState('stopped');
      this.emit('exit', { code, sig, stopping });
    });
  }

  scheduleRestart() {
    this.setState('down');
    const alive = Date.now() - this.lastStart;
    if (alive > 60000) this.restarts = 0;
    this.restarts++;
    const delay = Math.min(15000, 500 * Math.pow(2, Math.min(5, this.restarts - 1)));
    this.emit('log', `restarting engine in ${delay} ms`);
    clearTimeout(this.restartTimer);
    this.restartTimer = setTimeout(() => this.start(), delay);
  }

  setState(s) {
    if (this.state === s) return;
    this.state = s;
    if (s !== 'starting') this.loading = null;
    this.emit('state', s);
  }

  onLine(line) {
    const t = line.trim();
    if (!t) return;
    let ev;
    if (t[0] === '{') { try { ev = JSON.parse(t); } catch (_) { ev = null; } }
    if (!ev || typeof ev !== 'object' || !ev.ev) { this.emit('log', '[engine] ' + t); return; }
    if (ev.ev === 'ready') { this.setState('ready'); this.emit('ready'); return; }
    if (ev.ev === 'log') { this.emit('log', '[engine] ' + ev.msg); return; }
    if (ev.ev === 'loading' && (!ev.id || this.state === 'starting')) {
      this.loading = { what: ev.what, progress: ev.progress };
    }
    this.emit('event', ev);
  }

  send(obj) {
    if (!this.proc || !this.proc.stdin.writable) return false;
    this.proc.stdin.write(JSON.stringify(obj) + '\n');
    return true;
  }

  stop() {
    this.stopping = true;
    clearTimeout(this.restartTimer);
    if (this.proc) {
      try { this.proc.stdin.end(); } catch (_) { }
      const p = this.proc;
      setTimeout(() => { try { p.kill(); } catch (_) { } }, 1500).unref();
    }
  }

  // Stop the current process (if any) and start a fresh one.
  restart() {
    if (!this.proc) { clearTimeout(this.restartTimer); this.start(); return; }
    this.once('exit', () => this.start());
    this.stop();
  }

  status() {
    return { state: this.state, mock: this.mock, exe: this.exe, pid: this.pid, loading: this.loading };
  }
}

module.exports = { Engine };
