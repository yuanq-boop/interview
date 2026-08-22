import { spawn, type ChildProcess } from 'node:child_process';
import http from 'node:http';
import type { RepoInfo } from './repoAnalyzer.js';

export interface RunningServer {
  process: ChildProcess;
  url: string;
  stop: () => void;
}

/** Spawns the repo's dev script and blocks until it's actually serving HTTP. */
export async function startDevServer(repo: RepoInfo, explicitPort?: number): Promise<RunningServer> {
  const port = explicitPort ?? repo.guessedPort;
  const url = `http://localhost:${port}`;

  const [cmd, ...args] = repo.startCommand;
  const proc = spawn(cmd, args, {
    cwd: repo.root,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, BROWSER: 'none', PORT: String(port) },
  });

  const output: string[] = [];
  proc.stdout?.on('data', (d) => output.push(d.toString()));
  proc.stderr?.on('data', (d) => output.push(d.toString()));

  await waitForServer(url, 60_000, () => {
    if (proc.exitCode !== null) {
      throw new Error(`Dev server exited early (code ${proc.exitCode}):\n${output.join('')}`);
    }
  });

  return {
    process: proc,
    url,
    stop: () => proc.kill('SIGTERM'),
  };
}

function waitForServer(url: string, timeoutMs: number, checkAlive: () => void): Promise<void> {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      try {
        checkAlive();
      } catch (err) {
        reject(err);
        return;
      }
      const req = http.get(url, (res) => {
        res.resume();
        resolve();
      });
      req.on('error', () => {
        if (Date.now() - start > timeoutMs) {
          reject(new Error(`Timed out waiting for dev server at ${url}`));
        } else {
          setTimeout(tick, 500);
        }
      });
    };
    tick();
  });
}
