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

// Left pane is the UNTOUCHED base model served by OpenRouter -- deepseek-v4-flash-0731 is the
// exact checkpoint we fine-tuned, so this is a true A/B and not an approximation. Right pane is
// our trained model on local vllm. Hosting the base remotely leaves all 8 GPUs for ours, which
// matters because the fp8 copy is ~284 GB.
type Arm = { url: string; model: string; key: string; local: boolean };
const ARMS: Record<string, Arm> = {
  // Stock DeepSeek-V4-Flash, hosted, so the comparison is against the exact checkpoint we trained
  // from. Fireworks by default because the OpenRouter key hit its monthly limit; set
  // BASE_PROVIDER=openrouter to switch back once that resets.
  base:
    process.env.BASE_PROVIDER === "openrouter"
      ? {
          url: "https://openrouter.ai/api/v1",
          model: "deepseek/deepseek-v4-flash-0731",
          key: process.env.OPENROUTER_API_KEY ?? "",
          local: false,
        }
      : {
          url: "https://api.fireworks.ai/inference/v1",
          model: "accounts/fireworks/models/deepseek-v4-flash-0731",
          key: process.env.FIREWORKS_API_KEY ?? "",
          local: false,
        },
  tuned: {
    url: process.env.VLLM_URL ?? "http://127.0.0.1:8000/v1",
    model: "dsv4",
    key: "local",
    local: true,
  },
};
const armFor = (id: string): Arm => ARMS[id] ?? ARMS.base;
const DEFAULT_MODEL = "base";

// Just "Answer the question." Adding "If you don't know, say so" measurably suppressed tool
// use: with that clause the model answers "I cannot answer, that date is in the future" instead
// of searching (0 tool calls vs 1, verified against vllm directly). It told the model to admit
// ignorance rather than look it up.
const SYSTEM = "Answer the question.";

// DeepSeek's recommended sampling, not Qwen's. The previous presets (presence_penalty 1.5,
// top_k 20) came off the Qwen3.5 model card, and OpenRouter rejects vllm-only fields outright,
// so anything non-standard is sent only to the local server.
const PRESETS = {
  thinking: { temperature: 0.6, top_p: 0.95 },
  instruct: { temperature: 0.6, top_p: 0.95 },
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
 * DeepSeek-V4 has no chat template of its own -- vllm supplies one under --tokenizer-mode
 * deepseek_v4 -- so the flag that turns reasoning on is not Qwen's `enable_thinking`. Both
 * spellings plus OpenRouter's `reasoning` switch are sent; each provider ignores what it does
 * not recognise, which is cheaper than guessing wrong and silently grading non-thinking output.
 */
function provider(thinking: boolean, arm: Arm) {
  const preset = thinking ? PRESETS.thinking : PRESETS.instruct;
  return createOpenAICompatible({
    name: arm.local ? "vllm" : "openrouter",
    baseURL: arm.url,
    apiKey: arm.key,
    fetch: async (input, init) => {
      if (!init?.body) return fetch(input, init);
      const body: Record<string, unknown> = {
        ...JSON.parse(init.body as string),
        ...preset,
        chat_template_kwargs: { thinking, enable_thinking: thinking },
      };
      // Send-everything-and-let-them-ignore-it does not hold for Fireworks: it rejects an
      // unrecognised `reasoning` field outright with "Extra inputs are not permitted" rather
      // than dropping it, so the whole request fails. Only OpenRouter gets that switch.
      if (arm.url.includes("openrouter")) body.reasoning = { enabled: thinking };
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

  const arm = armFor(model);
  // The remote model advertises a 1M window; cap it so a runaway generation cannot bill
  // for one, and so both panes get comparable output budgets.
  const ctx = arm.local ? await contextWindow(arm.url) : 32768;

  const result = streamText({
    model: provider(thinking, arm).chatModel(arm.model),
    messages: await convertToModelMessages(messages),
    system: SYSTEM,
    // Offered only when the toggle is on. Nothing forces a call: whether the model searches, how
    // often, and with what query is entirely its own decision -- which is the point, since a
    // stale model choosing *not* to search is exactly the behaviour worth watching.
    tools: searchEnabled ? { webSearch } : undefined,
    stopWhen: stepCountIs(6),
    // Cap the completion rather than deriving it from the context window.
    //
    // maxOutputTokens is fixed once, before any tool has run, but the PROMPT grows every step as
    // search results accumulate. Reserving a flat slice of the window (ctx - 5000) was sized for a
    // 16k context; at 65k it asked for 60,536 completion tokens, so the step after a search sent
    // 11,225 prompt + 60,536 completion and the server rejected the whole request:
    //   Requested token count exceeds the model's maximum context length of 65536 tokens
    // The visible symptom is a model that searches, receives results and then stops -- the
    // follow-up never generates. A fixed ceiling leaves the rest of the window for prompt growth.
    maxOutputTokens: Math.min(8192, Math.max(512, Math.floor(ctx / 4))),
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
