import type { BrowserSession } from './browserSession.js';
import { Diagnostics } from './cdp/diagnostics.js';
import { pressKey, clickAt } from './cdp/input.js';
import { captureScreenshot } from './cdp/capture.js';
import type { TestPlan, TestStep, ScreenshotArtifact } from './types.js';

export interface ExecutionResult {
  artifacts: ScreenshotArtifact[];
  diagnostics: Diagnostics;
}

/** Runs a planner-generated TestPlan against a live BrowserSession via raw CDP calls. */
export async function executePlan(
  session: BrowserSession,
  plan: TestPlan,
  screenshotDir: string,
): Promise<ExecutionResult> {
  const diagnostics = new Diagnostics(session.cdp);
  await diagnostics.attach();

  const artifacts: ScreenshotArtifact[] = [];
  let lastStep: TestStep | undefined;

  for (const step of plan.steps) {
    switch (step.action) {
      case 'navigate':
        if (!step.url) throw new Error('navigate step missing "url"');
        await session.page.goto(step.url, { waitUntil: 'domcontentloaded' });
        break;

      case 'wait':
        await new Promise((resolve) => setTimeout(resolve, step.ms ?? 1000));
        break;

      case 'keypress':
        for (let i = 0; i < (step.times ?? 1); i++) {
          await pressKey(session.cdp, step.key ?? 'Space');
        }
        break;

      case 'click':
        await clickAt(session.cdp, step.x ?? 0, step.y ?? 0);
        break;

      case 'evaluate':
        if (step.expression) {
          await session.cdp.send('Runtime.evaluate', { expression: step.expression });
        }
        break;

      case 'screenshot': {
        const filePath = await captureScreenshot(session.cdp, screenshotDir, step.label ?? `screenshot-${artifacts.length + 1}`);
        const { consoleIssues, networkIssues } = diagnostics.drainSinceLastScreenshot();
        artifacts.push({
          label: step.label ?? `screenshot-${artifacts.length + 1}`,
          filePath,
          timestamp: Date.now(),
          precedingAction: lastStep,
          consoleIssuesSinceLast: consoleIssues,
          networkIssuesSinceLast: networkIssues,
        });
        break;
      }

      default:
        console.warn(`Unknown step action, skipping: ${JSON.stringify(step)}`);
    }
    lastStep = step;
  }

  return { artifacts, diagnostics };
}
