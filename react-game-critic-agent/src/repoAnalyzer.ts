import fs from 'node:fs';
import path from 'node:path';

export interface RepoInfo {
  root: string;
  name: string;
  packageManager: 'npm' | 'yarn' | 'pnpm';
  startScript: string;
  startCommand: string[];
  framework: 'vite' | 'cra' | 'next' | 'unknown';
  guessedPort: number;
  readme: string | null;
  sourceOverview: string;
}

/** Reads package.json + a shallow file listing to figure out how to boot the target game. */
export function analyzeRepo(root: string): RepoInfo {
  const pkgPath = path.join(root, 'package.json');
  if (!fs.existsSync(pkgPath)) {
    throw new Error(`No package.json found at ${pkgPath}. Is this a JS/React project?`);
  }
  const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf-8'));
  const deps = { ...pkg.dependencies, ...pkg.devDependencies };

  const packageManager: RepoInfo['packageManager'] = fs.existsSync(path.join(root, 'pnpm-lock.yaml'))
    ? 'pnpm'
    : fs.existsSync(path.join(root, 'yarn.lock'))
      ? 'yarn'
      : 'npm';

  const scripts: Record<string, string> = pkg.scripts || {};
  const startScript = scripts.dev ? 'dev' : scripts.start ? 'start' : Object.keys(scripts)[0];
  if (!startScript) {
    throw new Error('No "dev" or "start" script found in package.json.');
  }

  let framework: RepoInfo['framework'] = 'unknown';
  let guessedPort = 3000;
  if (deps.vite) {
    framework = 'vite';
    guessedPort = 5173;
  } else if (deps['react-scripts']) {
    framework = 'cra';
    guessedPort = 3000;
  } else if (deps.next) {
    framework = 'next';
    guessedPort = 3000;
  }

  const runCommand: Record<RepoInfo['packageManager'], string[]> = {
    npm: ['npm', 'run', startScript],
    yarn: ['yarn', startScript],
    pnpm: ['pnpm', 'run', startScript],
  };

  const readmePath = ['README.md', 'Readme.md', 'readme.md']
    .map((f) => path.join(root, f))
    .find((p) => fs.existsSync(p));

  return {
    root,
    name: pkg.name || path.basename(root),
    packageManager,
    startScript,
    startCommand: runCommand[packageManager],
    framework,
    guessedPort,
    readme: readmePath ? fs.readFileSync(readmePath, 'utf-8').slice(0, 4000) : null,
    sourceOverview: summarizeSourceTree(root),
  };
}

function summarizeSourceTree(root: string, maxEntries = 60): string {
  const ignored = new Set(['node_modules', '.git', 'dist', 'build', '.next', '.critic-agent']);
  const lines: string[] = [];

  function walk(dir: string, depth: number) {
    if (lines.length >= maxEntries || depth > 4) return;
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (ignored.has(entry.name)) continue;
      const full = path.join(dir, entry.name);
      const rel = path.relative(root, full);
      if (entry.isDirectory()) {
        lines.push(`${rel}/`);
        walk(full, depth + 1);
      } else if (/\.(jsx?|tsx?|css)$/.test(entry.name)) {
        lines.push(rel);
      }
      if (lines.length >= maxEntries) return;
    }
  }

  walk(root, 0);
  return lines.join('\n');
}
