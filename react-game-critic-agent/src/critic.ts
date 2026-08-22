import Anthropic from '@anthropic-ai/sdk';
import fs from 'node:fs';
import type { ScreenshotArtifact, CriticVerdict } from './types.js';

const CRITIC_SYSTEM_PROMPT = `You are a meticulous QA critic reviewing one screenshot from an automated
playtest of a browser-based React game. You'll be told what action was just performed and any
console/network errors captured since the previous screenshot. Judge ONLY what is visible plus
those error signals.

Look for: blank/white screens, error overlays (React error boundary, Vite/webpack overlay),
obviously broken layout (overlapping/invisible/off-screen elements), a game that appears
frozen/unresponsive to the preceding input, garbled text, or anything else a real player would
call a bug.

Respond with ONLY valid JSON, no prose, no markdown fences:
{"severity": "none"|"low"|"medium"|"high", "issue": "<short title or null>", "description": "<1-3 sentences>"}
"none" means it looks correct/expected for this point in the test.`;

export async function judgeScreenshot(artifact: ScreenshotArtifact): Promise<CriticVerdict> {
  const client = new Anthropic();
  const imageData = fs.readFileSync(artifact.filePath).toString('base64');

  const contextLines = [
    `Preceding action: ${JSON.stringify(artifact.precedingAction ?? 'initial load')}`,
    artifact.consoleIssuesSinceLast.length
      ? `Console issues since last screenshot: ${artifact.consoleIssuesSinceLast.map((c) => `[${c.type}] ${c.text}`).join(' | ')}`
      : 'No console issues since last screenshot.',
    artifact.networkIssuesSinceLast.length
      ? `Network issues since last screenshot: ${artifact.networkIssuesSinceLast.map((n) => `${n.url} ${n.status ?? n.errorText}`).join(' | ')}`
      : 'No network issues since last screenshot.',
  ].join('\n');

  const response = await client.messages.create({
    model: 'claude-sonnet-5',
    max_tokens: 500,
    system: CRITIC_SYSTEM_PROMPT,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: contextLines },
          { type: 'image', source: { type: 'base64', media_type: 'image/png', data: imageData } },
        ],
      },
    ],
  });

  const text = response.content
    .filter((block): block is Anthropic.TextBlock => block.type === 'text')
    .map((block) => block.text)
    .join('');

  const match = text.match(/\{[\s\S]*\}/);
  if (!match) {
    return {
      label: artifact.label,
      severity: 'none',
      issue: null,
      description: `Critic returned unparseable output: ${text}`,
    };
  }

  const parsed = JSON.parse(match[0]);
  return { label: artifact.label, severity: parsed.severity, issue: parsed.issue, description: parsed.description };
}

export async function judgeAll(artifacts: ScreenshotArtifact[]): Promise<CriticVerdict[]> {
  const verdicts: CriticVerdict[] = [];
  for (const artifact of artifacts) {
    verdicts.push(await judgeScreenshot(artifact));
  }
  return verdicts;
}
