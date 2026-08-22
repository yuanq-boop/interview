#!/usr/bin/env node
import 'dotenv/config';
import { Command } from 'commander';
import path from 'node:path';
import { analyzeRepo } from './repoAnalyzer.js';
import { startDevServer, type RunningServer } from './devServer.js';
import { BrowserSession } from './browserSession.js';
import { planTests } from './planner.js';
import { executePlan } from './executor.js';
import { judgeAll } from './critic.js';
import { buildReport, writeReport } from './report.js';

const program = new Command();

program
  .name('critic-agent')
  .description('AI critic agent that playtests a React game repo through a persistent Chrome/CDP session')
  .requiredOption('--repo <path>', 'path to the React game repo to test')
  .option('--url <url>', 'skip auto-starting a dev server and test this URL directly')
  .option('--port <port>', 'port to run/expect the dev server on', (v) => parseInt(v, 10))
  .option('--headless', 'run Chrome headless', false)
  .option('--profile <dir>', 'persistent Chrome profile directory', '.critic-agent/profile')
  .option('--out <dir>', 'output directory for screenshots and reports', '.critic-agent/reports')
  .option('--context <text>', 'extra hints for the planner (e.g. "controls are WASD, goal is to reach the flag")')
  .parse(process.argv);

const opts = program.opts<{
  repo: string;
  url?: string;
  port?: number;
  headless: boolean;
  profile: string;
  out: string;
  context?: string;
}>();

async function main(): Promise<void> {
  if (!process.env.ANTHROPIC_API_KEY) {
    throw new Error('ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.');
  }

  const repoRoot = path.resolve(opts.repo);
  const repo = analyzeRepo(repoRoot);
  console.log(`Analyzed repo "${repo.name}" (${repo.framework}), start script: ${repo.startScript}`);

  let devServer: RunningServer | null = null;
  let gameUrl = opts.url;

  if (!gameUrl) {
    console.log('Starting dev server...');
    devServer = await startDevServer(repo, opts.port);
    gameUrl = devServer.url;
    console.log(`Dev server up at ${gameUrl}`);
  }

  const runId = new Date().toISOString().replace(/[:.]/g, '-');
  const outDir = path.resolve(opts.out, runId);
  const profileDir = path.resolve(opts.profile);

  const session = await BrowserSession.launch(profileDir, opts.headless);

  try {
    console.log('Asking Claude to design a test plan...');
    const plan = await planTests(repo, gameUrl, opts.context);
    console.log(`Plan: ${plan.summary} (${plan.steps.length} steps)`);

    console.log('Executing plan via CDP...');
    const { artifacts, diagnostics } = await executePlan(session, plan, path.join(outDir, 'screenshots'));
    console.log(`Captured ${artifacts.length} screenshots.`);

    console.log('Asking the VLM critic to judge each screenshot...');
    const verdicts = await judgeAll(artifacts);

    const report = buildReport({
      repo: repo.name,
      url: gameUrl,
      planSummary: plan.summary,
      screenshots: artifacts,
      verdicts,
      diagnostics,
    });

    const { jsonPath, mdPath } = writeReport(report, outDir);
    console.log(`\nOverall verdict: ${report.overallVerdict.severity.toUpperCase()}`);
    console.log(report.overallVerdict.summary);
    console.log(`\nFull report: ${mdPath}`);
    console.log(`JSON report: ${jsonPath}`);
  } finally {
    await session.close();
    devServer?.stop();
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
