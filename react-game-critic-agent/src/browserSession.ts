import { chromium, type BrowserContext, type Page, type CDPSession } from 'playwright';
import fs from 'node:fs';

/**
 * A persistent Chrome session: same on-disk profile (cookies, localStorage, open state)
 * across separate `critic-agent` runs, plus a raw CDP session for low-level control
 * (input dispatch, screenshot capture, console/network monitoring) instead of
 * Playwright's higher-level page API.
 */
export class BrowserSession {
  private constructor(
    public context: BrowserContext,
    public page: Page,
    public cdp: CDPSession,
  ) {}

  static async launch(profileDir: string, headless = false): Promise<BrowserSession> {
    fs.mkdirSync(profileDir, { recursive: true });

    const launchOpts = {
      headless,
      viewport: { width: 1280, height: 800 },
      args: ['--disable-blink-features=AutomationControlled'],
    };

    if (process.env.CRITIC_CHROME_EXECUTABLE) {
      const context = await chromium.launchPersistentContext(profileDir, {
        ...launchOpts,
        executablePath: process.env.CRITIC_CHROME_EXECUTABLE,
      });
      const page = context.pages()[0] ?? (await context.newPage());
      const cdp = await context.newCDPSession(page);
      return new BrowserSession(context, page, cdp);
    }

    let context: BrowserContext;
    try {
      context = await chromium.launchPersistentContext(profileDir, {
        ...launchOpts,
        channel: process.env.CRITIC_CHROME_CHANNEL || 'chrome',
      });
    } catch (err) {
      console.warn(`System Chrome unavailable, falling back to bundled Chromium (${(err as Error).message})`);
      context = await chromium.launchPersistentContext(profileDir, launchOpts);
    }

    const page = context.pages()[0] ?? (await context.newPage());
    const cdp = await context.newCDPSession(page);

    return new BrowserSession(context, page, cdp);
  }

  async close(): Promise<void> {
    await this.context.close();
  }
}
