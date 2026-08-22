import type { CDPSession } from 'playwright';

/**
 * Raw CDP input dispatch (Input.dispatchKeyEvent / Input.dispatchMouseEvent) so keystrokes
 * and clicks land on the page the same way real hardware input would, independent of
 * Playwright's element-targeting locator API.
 */
const KEY_CODES: Record<string, { code: string; keyCode: number }> = {
  ArrowUp: { code: 'ArrowUp', keyCode: 38 },
  ArrowDown: { code: 'ArrowDown', keyCode: 40 },
  ArrowLeft: { code: 'ArrowLeft', keyCode: 37 },
  ArrowRight: { code: 'ArrowRight', keyCode: 39 },
  Space: { code: 'Space', keyCode: 32 },
  Enter: { code: 'Enter', keyCode: 13 },
  Escape: { code: 'Escape', keyCode: 27 },
  w: { code: 'KeyW', keyCode: 87 },
  a: { code: 'KeyA', keyCode: 65 },
  s: { code: 'KeyS', keyCode: 83 },
  d: { code: 'KeyD', keyCode: 68 },
};

export async function pressKey(cdp: CDPSession, key: string, holdMs = 60): Promise<void> {
  const mapped = KEY_CODES[key] ?? { code: key, keyCode: key.toUpperCase().charCodeAt(0) };
  const base = {
    key,
    code: mapped.code,
    windowsVirtualKeyCode: mapped.keyCode,
    nativeVirtualKeyCode: mapped.keyCode,
  };
  await cdp.send('Input.dispatchKeyEvent', { type: 'keyDown', ...base });
  await new Promise((resolve) => setTimeout(resolve, holdMs));
  await cdp.send('Input.dispatchKeyEvent', { type: 'keyUp', ...base });
}

export async function clickAt(cdp: CDPSession, x: number, y: number): Promise<void> {
  await cdp.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y });
  await cdp.send('Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button: 'left', clickCount: 1 });
  await cdp.send('Input.dispatchMouseEvent', { type: 'mouseReleased', x, y, button: 'left', clickCount: 1 });
}
