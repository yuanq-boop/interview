#!/usr/bin/env node
/**
 * Scripted bot playtest. Drives the game through the same input path a player
 * uses — look deltas, trigger, reload — and reports whether the core loop
 * actually functions: enemies spawn, shots register, waves advance, the player
 * can die. Screenshots are captured at fixed beats for visual evidence.
 *
 * Usage:
 *   node scripts/playtest.mjs --url http://127.0.0.1:5188 --seconds 90 --out artifacts/playtest
 */
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

function parseArgs(argv) {
  const args = {
    url: 'http://127.0.0.1:5188',
    seconds: 90,
    out: 'artifacts/playtest',
    seed: 12345,
    mobile: false,
    godMode: false,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (flag === '--url') (args.url = required(flag, value)), (i += 1);
    else if (flag === '--seconds') (args.seconds = Number(required(flag, value))), (i += 1);
    else if (flag === '--out') (args.out = required(flag, value)), (i += 1);
    else if (flag === '--seed') (args.seed = Number(required(flag, value))), (i += 1);
    else if (flag === '--mobile') args.mobile = true;
    else if (flag === '--god') args.godMode = true;
    else throw new Error(`Unknown argument: ${flag}`);
  }
  return args;
}

function required(flag, value) {
  if (value === undefined || value.startsWith('--')) throw new Error(`${flag} needs a value`);
  return value;
}

async function main() {
  const args = parseArgs(process.argv);
  const { chromium, devices } = await import('@playwright/test');
  const executablePath = process.env.CHROMIUM_EXECUTABLE_PATH;

  const browser = await chromium.launch(
    executablePath ? { executablePath } : { channel: 'chromium' },
  );
  const context = await browser.newContext(
    args.mobile
      ? { ...devices['Pixel 7'], hasTouch: true, isMobile: true }
      : { viewport: { width: 1280, height: 720 } },
  );
  const page = await context.newPage();

  const consoleErrors = [];
  const pageErrors = [];
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });
  page.on('pageerror', (error) => pageErrors.push(String(error)));

  await page.goto(args.url, { waitUntil: 'load' });
  await page.waitForFunction(() => Boolean(window.__THREE_GAME_TEST_HOOKS__), null, {
    timeout: 20000,
  });

  await page.evaluate((seed) => window.__THREE_GAME_TEST_HOOKS__.seed(seed), args.seed);
  await page.click('#overlay-action');
  await page.waitForFunction(
    () => window.__THREE_GAME_DIAGNOSTICS__?.phase === 'playing',
    null,
    { timeout: 10000 },
  );

  await mkdir(args.out, { recursive: true });
  const samples = [];
  const shots = [];
  const started = Date.now();
  let screenshotIndex = 0;
  let nextScreenshot = 6;

  // Aim/fire loop. Each tick reads the nearest enemy from diagnostics, turns
  // toward it with a bounded look delta, and holds the trigger when roughly on
  // target — the same path a human's mouse and mouse button feed.
  while ((Date.now() - started) / 1000 < args.seconds) {
    const tick = await page.evaluate((god) => {
      const diagnostics = window.__THREE_GAME_DIAGNOSTICS__;
      const hooks = window.__THREE_GAME_TEST_HOOKS__;
      if (!diagnostics || !hooks?.botInput) return null;

      const target = diagnostics.nearestEnemy;
      const player = diagnostics.player;
      let lookX = 0;
      let lookY = 0;
      let onTarget = false;

      if (target) {
        const dx = target.x - player.position.x;
        const dz = target.z - player.position.z;
        const horizontal = Math.hypot(dx, dz);
        const desiredYaw = Math.atan2(-dx, -dz);
        const desiredPitch = Math.atan2(target.y - (player.position.y + 1.68), horizontal);

        let yawError = desiredYaw - player.yaw;
        yawError = ((yawError + Math.PI) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2) - Math.PI;
        const pitchError = desiredPitch - player.pitch;

        // Look deltas are divided by sensitivity because botInput feeds raw
        // device units, exactly like a mouse would. Both axes are negated: the
        // game subtracts look delta, so turning left needs a negative delta.
        const sensitivity = 0.0022;
        lookX = Math.max(-400, Math.min(400, (-yawError / sensitivity) * 0.45));
        lookY = Math.max(-400, Math.min(400, (-pitchError / sensitivity) * 0.45));
        // Only shoot when the aim is inside roughly half a torso width at this
        // range, so accuracy measures the game rather than the bot's spraying.
        const tolerance = Math.max(0.01, Math.atan2(0.45, Math.max(1, target.distance)));
        onTarget = Math.abs(yawError) < tolerance && Math.abs(pitchError) < tolerance;
      }

      const empty = diagnostics.ammo.mag === 0;
      hooks.botInput({
        move: { x: 0, y: 0 },
        look: { x: lookX, y: lookY },
        // Never shoot through cover: without this the bot empties magazines
        // into walls and the accuracy figure means nothing.
        fire: onTarget && !empty && Boolean(target && target.visible),
        ads: Boolean(target) && target.distance > 6,
        reload: empty,
      });
      if (god !== undefined) void god;

      return {
        elapsed: diagnostics.elapsed,
        phase: diagnostics.phase,
        wave: diagnostics.wave,
        kills: diagnostics.kills,
        headshots: diagnostics.headshots,
        accuracy: diagnostics.accuracy,
        health: diagnostics.player.health,
        alive: diagnostics.player.alive,
        enemiesAlive: diagnostics.enemiesAlive,
        ammo: diagnostics.ammo.mag,
        reserve: diagnostics.ammo.reserve,
        score: diagnostics.score,
        targetDistance: target ? target.distance : null,
      };
    }, args.godMode || undefined);

    if (tick) {
      samples.push(tick);
      shots.push(tick.kills);
    }

    const seconds = (Date.now() - started) / 1000;
    if (seconds >= nextScreenshot) {
      screenshotIndex += 1;
      nextScreenshot += 14;
      const file = path.join(
        args.out,
        `${args.mobile ? 'mobile' : 'desktop'}-playtest-${String(screenshotIndex).padStart(2, '0')}.png`,
      );
      await page.screenshot({ path: file });
    }

    if (tick && tick.phase === 'dead') break;
    await page.waitForTimeout(60);
  }

  const last = samples.at(-1) ?? null;
  const first = samples[0] ?? null;
  const maxWave = samples.reduce((max, sample) => Math.max(max, sample.wave), 0);
  const report = {
    url: args.url,
    seed: args.seed,
    mobile: args.mobile,
    durationSeconds: Number(((Date.now() - started) / 1000).toFixed(1)),
    samples: samples.length,
    first,
    last,
    maxWave,
    totalKills: last ? last.kills : 0,
    headshots: last ? last.headshots : 0,
    accuracy: last ? Number((last.accuracy * 100).toFixed(1)) : 0,
    diedDuringRun: samples.some((sample) => sample.phase === 'dead'),
    everTookDamage: samples.some((sample) => sample.health < 100),
    everReloaded: samples.some((sample) => sample.ammo === 0),
    screenshots: screenshotIndex,
    consoleErrors,
    pageErrors,
  };

  await writeFile(
    path.join(args.out, `${args.mobile ? 'mobile' : 'desktop'}-playtest.json`),
    `${JSON.stringify(report, null, 2)}\n`,
  );
  console.log(JSON.stringify(report, null, 2));

  await browser.close();

  if (pageErrors.length > 0) {
    console.error('Page errors detected during playtest.');
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
