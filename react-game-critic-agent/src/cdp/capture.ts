import type { CDPSession } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

/** Screenshot capture via the raw CDP Page domain rather than page.screenshot(). */
export async function captureScreenshot(cdp: CDPSession, outDir: string, label: string): Promise<string> {
  const { data } = await cdp.send('Page.captureScreenshot', { format: 'png' });
  fs.mkdirSync(outDir, { recursive: true });
  const safeLabel = label.replace(/[^a-z0-9-_]/gi, '-');
  const filePath = path.join(outDir, `${Date.now()}-${safeLabel}.png`);
  fs.writeFileSync(filePath, Buffer.from(data, 'base64'));
  return filePath;
}
