import Anthropic from '@anthropic-ai/sdk';
import type { RepoInfo } from './repoAnalyzer.js';
import type { TestPlan } from './types.js';

const PLAN_SYSTEM_PROMPT = `You are a QA engineer designing a black-box playtest for a browser-based React game.
You will be given the game's package.json summary, README (if any), and a source file listing.
Produce a JSON test plan: a short sequence of browser actions that load the game and probe for
common bugs (broken rendering, unresponsive controls, crashes on interaction, dead-end states).

Return ONLY valid JSON matching this shape, no prose, no markdown fences:
{
  "summary": "one sentence describing what you'll test",
  "steps": [
    {"action":"navigate","url":"<the game url>"},
    {"action":"wait","ms":1500},
    {"action":"screenshot","label":"initial-load"},
    {"action":"keypress","key":"ArrowUp","times":3},
    {"action":"wait","ms":300},
    {"action":"screenshot","label":"after-input"},
    {"action":"click","x":640,"y":400},
    {"action":"screenshot","label":"after-click"}
  ]
}

Valid actions: navigate, wait, screenshot, keypress (key + optional times), click (x + y),
evaluate (expression: a JS snippet run in-page).

Keep the plan to 10-18 steps: load the game, then probe whichever controls the source suggests
(arrow keys, WASD, space, mouse clicks), interleaving screenshots so a human/VLM can compare
before/after each probe. Include at least one screenshot after leaving the game idle for a few
seconds (animation loops, memory leaks) and at least one attempt at a likely edge case (rapid
repeated input, an out-of-bounds click, restarting after a loss/win if applicable).`;

export async function planTests(repo: RepoInfo, gameUrl: string, extraContext?: string): Promise<TestPlan> {
  const client = new Anthropic();

  const userPrompt = `Game URL to navigate to: ${gameUrl}

Package: ${repo.name}
Framework: ${repo.framework}
Start script: ${repo.startScript}

README excerpt:
${repo.readme ?? '(none found)'}

Source file listing:
${repo.sourceOverview}
${extraContext ? `\nAdditional context from the user:\n${extraContext}` : ''}`;

  const response = await client.messages.create({
    model: 'claude-sonnet-5',
    max_tokens: 2000,
    system: PLAN_SYSTEM_PROMPT,
    messages: [{ role: 'user', content: userPrompt }],
  });

  const text = response.content
    .filter((block): block is Anthropic.TextBlock => block.type === 'text')
    .map((block) => block.text)
    .join('');

  return parsePlan(text);
}

function parsePlan(text: string): TestPlan {
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) throw new Error(`Planner did not return JSON:\n${text}`);
  const plan = JSON.parse(match[0]) as TestPlan;
  if (!Array.isArray(plan.steps) || plan.steps.length === 0) {
    throw new Error('Planner returned an empty or invalid step list.');
  }
  return plan;
}
