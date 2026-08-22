import fs from 'node:fs';
import path from 'node:path';
import type { FinalReport, CriticVerdict, ScreenshotArtifact, Severity } from './types.js';
import type { Diagnostics } from './cdp/diagnostics.js';

const SEVERITY_RANK: Record<Severity, number> = { none: 0, low: 1, medium: 2, high: 3 };

export function buildReport(params: {
  repo: string;
  url: string;
  planSummary: string;
  screenshots: ScreenshotArtifact[];
  verdicts: CriticVerdict[];
  diagnostics: Diagnostics;
}): FinalReport {
  const worst = params.verdicts.reduce<CriticVerdict>(
    (acc, v) => (SEVERITY_RANK[v.severity] > SEVERITY_RANK[acc.severity] ? v : acc),
    { severity: 'none', issue: null, label: '', description: '' },
  );

  return {
    repo: params.repo,
    url: params.url,
    planSummary: params.planSummary,
    screenshots: params.screenshots,
    verdicts: params.verdicts,
    overallVerdict: {
      severity: worst.severity,
      summary:
        worst.severity === 'none'
          ? 'No issues detected across the playtest sequence.'
          : `Most severe finding (${worst.severity}) at "${worst.label}": ${worst.issue} — ${worst.description}`,
    },
    allConsoleIssues: params.diagnostics.allConsoleIssues,
    allNetworkIssues: params.diagnostics.allNetworkIssues,
    generatedAt: new Date().toISOString(),
  };
}

export function writeReport(report: FinalReport, outDir: string): { jsonPath: string; mdPath: string } {
  fs.mkdirSync(outDir, { recursive: true });
  const jsonPath = path.join(outDir, 'report.json');
  const mdPath = path.join(outDir, 'report.md');

  fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2));
  fs.writeFileSync(mdPath, renderMarkdown(report, outDir));

  return { jsonPath, mdPath };
}

function renderMarkdown(report: FinalReport, outDir: string): string {
  const lines: string[] = [];
  lines.push('# Critic Agent Report', '');
  lines.push(`- **Repo:** ${report.repo}`);
  lines.push(`- **URL:** ${report.url}`);
  lines.push(`- **Generated:** ${report.generatedAt}`);
  lines.push(`- **Test plan:** ${report.planSummary}`);
  lines.push(`- **Overall verdict:** ${report.overallVerdict.severity.toUpperCase()} — ${report.overallVerdict.summary}`);
  lines.push('', '## Steps', '');

  for (const shot of report.screenshots) {
    const verdict = report.verdicts.find((v) => v.label === shot.label);
    const relPath = path.relative(outDir, shot.filePath);
    lines.push(`### ${shot.label}`, '');
    lines.push(`![${shot.label}](${relPath})`, '');
    if (shot.precedingAction) lines.push(`- Preceding action: \`${JSON.stringify(shot.precedingAction)}\``);
    if (verdict) lines.push(`- Critic verdict: **${verdict.severity}** — ${verdict.issue ?? 'no issue'}: ${verdict.description}`);
    if (shot.consoleIssuesSinceLast.length) {
      lines.push(`- Console issues: ${shot.consoleIssuesSinceLast.map((c) => c.text).join('; ')}`);
    }
    if (shot.networkIssuesSinceLast.length) {
      lines.push(`- Network issues: ${shot.networkIssuesSinceLast.map((n) => `${n.url} (${n.status ?? n.errorText})`).join('; ')}`);
    }
    lines.push('');
  }

  if (report.allConsoleIssues.length || report.allNetworkIssues.length) {
    lines.push('## Raw diagnostics', '');
    if (report.allConsoleIssues.length) {
      lines.push(`### Console (${report.allConsoleIssues.length})`);
      for (const c of report.allConsoleIssues) lines.push(`- [${c.type}] ${c.text}`);
      lines.push('');
    }
    if (report.allNetworkIssues.length) {
      lines.push(`### Network (${report.allNetworkIssues.length})`);
      for (const n of report.allNetworkIssues) lines.push(`- ${n.url} — ${n.status ?? n.errorText}`);
    }
  }

  return lines.join('\n');
}
