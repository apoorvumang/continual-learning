/**
 * Keenable web search, server-side only.
 *
 * The API key must never reach the browser, so this module is imported exclusively from route
 * handlers. Results are cached in-process by query string: two models asked the same question
 * usually issue overlapping queries, and a side-by-side comparison is only fair if both see
 * identical results.
 */

const BASE = "https://api.keenable.ai";
const MAX_RPS = 4; // documented ceiling is 10/s per org; stay well under it

let nextSlot = 0;
async function throttle() {
  const gap = 1000 / MAX_RPS;
  const now = Date.now();
  const wait = Math.max(0, nextSlot - now);
  nextSlot = Math.max(now, nextSlot) + gap;
  if (wait > 0) await new Promise((r) => setTimeout(r, wait));
}

export type Hit = { title: string; text: string; url: string };

const cache = new Map<string, Hit[]>();

export async function search(query: string, topk = 5): Promise<Hit[]> {
  const key = `${query}::${topk}`;
  const hit = cache.get(key);
  if (hit) return hit;

  const apiKey = process.env.KEENABLE_API_KEY;
  if (!apiKey) throw new Error("KEENABLE_API_KEY is not set on the server");

  await throttle();
  const res = await fetch(`${BASE}/v1/search`, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) throw new Error(`keenable ${res.status}: ${(await res.text()).slice(0, 200)}`);

  const json = (await res.json()) as { results?: Array<Record<string, string>> };
  const hits: Hit[] = (json.results ?? []).slice(0, topk).map((r) => ({
    title: r.title ?? "",
    text: r.snippet ?? r.description ?? "",
    url: r.url ?? "",
  }));
  cache.set(key, hits);
  return hits;
}

/**
 * Let the model choose its own queries, then hand back what it found.
 *
 * Deliberately the model's own queries rather than the user's text: what a stale model chooses
 * to search for is the interesting signal. Asked why Dubai flights were cheap, one checkpoint
 * searched "Air India price increase 2026", retrieved fare-aggregator pages, and then argued
 * against the real cause -- which is visible only if the queries are its own and are shown.
 */
export async function research(
  ask: (prompt: string, history: Array<{ role: "assistant" | "user"; content: string }>) => Promise<string>,
  question: string,
  maxSearches = 4
): Promise<{ queries: string[]; notes: string }> {
  const system = `Search the web to answer the question.

Reply with exactly one line each turn:
SEARCH: <query>
DONE

Up to ${maxSearches} searches. Reply DONE when ready.`;

  const history: Array<{ role: "assistant" | "user"; content: string }> = [];
  const queries: string[] = [];
  const chunks: string[] = [];

  for (let i = 0; i <= maxSearches; i++) {
    const out = (await ask(system, [{ role: "user", content: question }, ...history])).trim();
    const m = out.match(/SEARCH:\s*(.+)/);
    if (!m || queries.length >= maxSearches) break;
    const q = m[1].trim();
    queries.push(q);
    let obs: string;
    try {
      const hits = await search(q);
      obs = hits.map((h) => `[${h.title}] ${h.text}`).join("\n\n") || "No results.";
    } catch (e) {
      obs = `Search failed: ${e instanceof Error ? e.message : String(e)}`;
    }
    chunks.push(`### ${q}\n${obs}`);
    history.push({ role: "assistant", content: out });
    history.push({ role: "user", content: `Results:\n${obs}` });
  }
  return { queries, notes: chunks.join("\n\n") };
}
