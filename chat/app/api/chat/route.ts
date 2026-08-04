import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import {
  convertToModelMessages,
  createUIMessageStreamResponse,
  stepCountIs,
  streamText,
  tool,
  toUIMessageStream,
  type UIMessage,
} from "ai";
import { z } from "zod";

import { search } from "@/lib/search";

// Each arm is its own vllm process with MERGED weights, not one process with a LoRA adapter.
// Serving the adapter was non-deterministic at temperature 0 -- identical requests returned
// different answers, while the un-adapted arm on the same server was stable. See chat/README.md.
const ENDPOINTS: Record<string, string> = {
  stock: process.env.VLLM_STOCK_URL ?? "http://127.0.0.1:8010/v1",
  armP: process.env.VLLM_ARMP_URL ?? "http://127.0.0.1:8011/v1",
};
const DEFAULT_MODEL = process.env.VLLM_MODEL ?? "stock";
const urlFor = (model: string) => ENDPOINTS[model] ?? ENDPOINTS[DEFAULT_MODEL];

// Just "Answer the question." Adding "If you don't know, say so" measurably suppressed tool
// use: with that clause the model answers "I cannot answer, that date is in the future" instead
// of searching (0 tool calls vs 1, verified against vllm directly). It told the model to admit
// ignorance rather than look it up.
const SYSTEM = "Answer the question.";

// Sampling presets straight from the Qwen3.5 model card. Qwen3.5 thinks by default and has no
// /nothink soft switch, so the mode is selected with chat_template_kwargs.
const PRESETS = {
  thinking: { temperature: 1.0, top_p: 0.95, presence_penalty: 1.5 },
  instruct: { temperature: 0.7, top_p: 0.8, presence_penalty: 1.5 },
} as const;

// Output tokens share one window with the prompt, so a fixed budget silently breaks whenever a
// server is started with a smaller --max-model-len. Ask the server instead of hardcoding it.
const ctxCache = new Map<string, number>();
async function contextWindow(baseUrl: string): Promise<number> {
  const cached = ctxCache.get(baseUrl);
  if (cached) return cached;
  let len = 8192;
  try {
    const res = await fetch(`${baseUrl}/models`, { headers: { Authorization: "Bearer local" } });
    const json = (await res.json()) as { data?: Array<{ max_model_len?: number }> };
    len = json.data?.[0]?.max_model_len ?? len;
  } catch {
    // fall through to the conservative default
  }
  ctxCache.set(baseUrl, len);
  return len;
}

/**
 * top_k, min_p and chat_template_kwargs are vllm extensions with no slot in the AI SDK's model
 * settings, so they are merged into the outgoing JSON body in a custom fetch.
 */
function vllm(thinking: boolean, model: string) {
  const preset = thinking ? PRESETS.thinking : PRESETS.instruct;
  return createOpenAICompatible({
    name: "vllm",
    baseURL: urlFor(model),
    apiKey: "local",
    fetch: async (input, init) => {
      if (!init?.body) return fetch(input, init);
      const body = {
        ...JSON.parse(init.body as string),
        ...preset,
        top_k: 20,
        min_p: 0,
        chat_template_kwargs: { enable_thinking: thinking },
      };
      return fetch(input, { ...init, body: JSON.stringify(body) });
    },
  });
}

/** An ordinary tool. The model calls it if it wants to, when it wants to, or not at all. */
const webSearch = tool({
  description: "Search the web for current information.",
  inputSchema: z.object({ query: z.string().describe("Search query") }),
  execute: async ({ query }) => {
    const hits = await search(query);
    return hits.map((h) => ({ title: h.title, snippet: h.text, url: h.url }));
  },
});

export async function POST(req: Request) {
  const {
    messages,
    thinking = false,
    model = DEFAULT_MODEL,
    search: searchEnabled = false,
  }: {
    messages: UIMessage[];
    thinking?: boolean;
    model?: string;
    search?: boolean;
  } = await req.json();

  const ctx = await contextWindow(urlFor(model));

  const result = streamText({
    model: vllm(thinking, model).chatModel(model),
    messages: await convertToModelMessages(messages),
    system: SYSTEM,
    // Offered only when the toggle is on. Nothing forces a call: whether the model searches, how
    // often, and with what query is entirely its own decision -- which is the point, since a
    // stale model choosing *not* to search is exactly the behaviour worth watching.
    tools: searchEnabled ? { webSearch } : undefined,
    stopWhen: stepCountIs(6),
    // Tool results accumulate across steps, so leave more room when the tool is available.
    maxOutputTokens: Math.max(512, ctx - (searchEnabled ? 5000 : 1200)),
  });

  return createUIMessageStreamResponse({
    stream: toUIMessageStream({
      stream: result.stream,
      sendReasoning: true,
      // The default is `() => "An error occurred."`, which hides the cause on purpose. This is an
      // internal tool on a tailnet and the upstream message is usually the whole diagnosis -- a
      // thinking request against a too-small --max-model-len looked for a while like thinking
      // itself being broken.
      onError: (error) => {
        console.error("[chat] upstream error:", error);
        return error instanceof Error ? error.message : String(error);
      },
    }),
  });
}
