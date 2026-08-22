export type StepAction = 'navigate' | 'wait' | 'screenshot' | 'keypress' | 'click' | 'evaluate';

export interface TestStep {
  action: StepAction;
  url?: string;
  ms?: number;
  label?: string;
  key?: string;
  times?: number;
  x?: number;
  y?: number;
  expression?: string;
  note?: string;
}

export interface TestPlan {
  summary: string;
  steps: TestStep[];
}

export type Severity = 'none' | 'low' | 'medium' | 'high';

export interface ConsoleIssue {
  type: 'error' | 'warning' | 'exception';
  text: string;
  timestamp: number;
}

export interface NetworkIssue {
  url: string;
  status?: number;
  errorText?: string;
  timestamp: number;
}

export interface ScreenshotArtifact {
  label: string;
  filePath: string;
  timestamp: number;
  precedingAction?: TestStep;
  consoleIssuesSinceLast: ConsoleIssue[];
  networkIssuesSinceLast: NetworkIssue[];
}

export interface CriticVerdict {
  label: string;
  severity: Severity;
  issue: string | null;
  description: string;
}

export interface FinalReport {
  repo: string;
  url: string;
  planSummary: string;
  screenshots: ScreenshotArtifact[];
  verdicts: CriticVerdict[];
  overallVerdict: {
    severity: Severity;
    summary: string;
  };
  allConsoleIssues: ConsoleIssue[];
  allNetworkIssues: NetworkIssue[];
  generatedAt: string;
}
