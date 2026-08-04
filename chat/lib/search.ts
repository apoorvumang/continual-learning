/**
 * Keenable web search, server-side only.
 *
 * The API key must never reach the browser, so this module is imported exclusively from route
 * handlers. Results are cached in-process by query string: two models asked the same question
 * often issue overlapping queries, and a side-by-side comparison is only fair if both see
 * identical results for an identical query.
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
