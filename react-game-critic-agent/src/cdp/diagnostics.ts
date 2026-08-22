import type { CDPSession } from 'playwright';
import type { ConsoleIssue, NetworkIssue } from '../types.js';

/**
 * Objective, non-VLM signal collected straight from the CDP Runtime/Log/Network domains:
 * console errors/warnings, uncaught exceptions, and failed/4xx+/5xx network requests.
 * These are attached alongside each screenshot so the critic isn't judging pixels alone.
 */
export class Diagnostics {
  allConsoleIssues: ConsoleIssue[] = [];
  allNetworkIssues: NetworkIssue[] = [];

  private pendingConsole: ConsoleIssue[] = [];
  private pendingNetwork: NetworkIssue[] = [];

  constructor(private cdp: CDPSession) {}

  async attach(): Promise<void> {
    await this.cdp.send('Runtime.enable');
    await this.cdp.send('Network.enable');

    this.cdp.on('Runtime.consoleAPICalled', (event: any) => {
      if (event.type !== 'error' && event.type !== 'warning') return;
      this.record({
        type: event.type === 'error' ? 'error' : 'warning',
        text: event.args.map((a: any) => a.value ?? a.description ?? '').join(' '),
        timestamp: Date.now(),
      });
    });

    this.cdp.on('Runtime.exceptionThrown', (event: any) => {
      this.record({
        type: 'exception',
        text: event.exceptionDetails?.exception?.description || event.exceptionDetails?.text || 'Unknown exception',
        timestamp: Date.now(),
      });
    });

    this.cdp.on('Network.responseReceived', (event: any) => {
      if (event.response.status < 400) return;
      this.recordNetwork({
        url: event.response.url,
        status: event.response.status,
        timestamp: Date.now(),
      });
    });

    this.cdp.on('Network.loadingFailed', (event: any) => {
      this.recordNetwork({
        url: event.requestId,
        errorText: event.errorText,
        timestamp: Date.now(),
      });
    });
  }

  private record(issue: ConsoleIssue): void {
    this.allConsoleIssues.push(issue);
    this.pendingConsole.push(issue);
  }

  private recordNetwork(issue: NetworkIssue): void {
    this.allNetworkIssues.push(issue);
    this.pendingNetwork.push(issue);
  }

  /** Returns and clears everything captured since the last screenshot. */
  drainSinceLastScreenshot(): { consoleIssues: ConsoleIssue[]; networkIssues: NetworkIssue[] } {
    const result = { consoleIssues: this.pendingConsole, networkIssues: this.pendingNetwork };
    this.pendingConsole = [];
    this.pendingNetwork = [];
    return result;
  }
}
