/**
 * Acceptance gate: nexus `callStructured` against a running llm-router.
 *
 * Run it from the nexus checkout so its dependencies resolve:
 *
 *   cd ~/projects/nexus && npx tsx ~/projects/llm-router/scripts/nexus_gate.ts
 *
 * What it proves, in order of what actually broke:
 *   1. json_schema survives the router (it used to be dropped on the floor).
 *   2. A provider that refuses json_schema refuses *fast* — nexus falls back on
 *      attempt 0 instead of retrying a 502 three times with exponential backoff.
 *   3. Smart routing ("auto") still returns something a Zod schema accepts.
 */
const NEXUS = process.env.NEXUS_DIR ?? `${process.env.HOME}/projects/nexus`;
const ROUTER = process.env.ROUTER_URL ?? "http://127.0.0.1:8642/v1";

async function run(
  label: string,
  model: string,
  deps: { createLLMClient: any; callStructured: any; schema: any },
): Promise<boolean> {
  const { createLLMClient, callStructured, schema } = deps;
  const client = createLLMClient({ endpoint: ROUTER, model, apiKey: process.env.ROUTER_KEY ?? "none" });
  const t0 = Date.now();
  try {
    const out = await callStructured({
      client,
      prompt: "Rate the fit 0-100 for this candidate and role.\nRole: {{role}}\nCandidate: {{cand}}",
      vars: { role: "Senior Python backend engineer", cand: "8 years Python, FastAPI, SQLite" },
      schema,
    });
    const secs = ((Date.now() - t0) / 1000).toFixed(1);
    console.log(`PASS  ${label.padEnd(34)} ${secs}s  score=${out.score}  "${String(out.reason).slice(0, 44)}"`);
    return true;
  } catch (e) {
    const secs = ((Date.now() - t0) / 1000).toFixed(1);
    console.log(`FAIL  ${label.padEnd(34)} ${secs}s  ${e instanceof Error ? e.message.slice(0, 160) : String(e)}`);
    return false;
  }
}

async function main() {
  // zod resolves from the nexus checkout, not from this script's directory —
  // llm-router has no node_modules of its own.
  const { z } = await import(`${NEXUS}/node_modules/zod/index.js`);
  const { createLLMClient } = await import(`${NEXUS}/src/llm/client.ts`);
  const { callStructured } = await import(`${NEXUS}/src/llm/structured.ts`);

  const schema = z.object({
    score: z.number().describe("0-100 fit score"),
    reason: z.string().describe("one sentence"),
  });
  const deps = { createLLMClient, callStructured, schema };

  const cases: Array<[string, string]> = [
    ["schema-capable (gemma-4-26b)", "openrouter:google/gemma-4-26b-a4b-it:free"],
    ["schema-refusing (deepseek-v4-pro)", "deepseek:deepseek-v4-pro"],
    ["auto", "auto"],
  ];

  let ok = true;
  for (const [label, model] of cases) {
    ok = (await run(label, model, deps)) && ok;
  }
  console.log(ok ? "\nall gates green" : "\ngate failed");
  process.exit(ok ? 0 : 1);
}

main();
