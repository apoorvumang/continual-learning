import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import {
  convertToModelMessages,
  createUIMessageStreamResponse,
  streamText,
  toUIMessageStream,
  type UIMessage,
} from "ai";

import { research } from "@/lib/search";

// Each arm is its own vllm process with MERGED weights, not one process with a LoRA adapter.
// Serving the adapter was non-deterministic at temperature 0 -- identical requests returned
// different answers, while the un-adapted arm on the same server was stable -- so the adapter
// path cannot be trusted for a comparison. See chat/README.md.
const ENDPOINTS: Record<string, string> = {
  stock: process.env.VLLM_STOCK_URL ?? "http://127.0.0.1:8010/v1",
  armP: process.env.VLLM_ARMP_URL ?? "http://127.0.0.1:8011/v1",
};
const DEFAULT_MODEL = process.env.VLLM_MODEL ?? "stock";
const urlFor = (model: string) => ENDPOINTS[model] ?? ENDPOINTS[DEFAULT_MODEL];

// Output tokens share one window with the prompt, so a fixed thinking budget silently breaks
// whenever the server is started with a smaller --max-model-len. That happened twice: once at
// --max-model-len 8192 with an 8192 budget, and again when serving two 35B models forced the
// context down to fit in memory. So ask the server what its window is and size the budget to
// fit, rather than hardcoding a number that has to agree with a flag somewhere else.
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

/** Leave room for the prompt: research notes are capped at 14k chars, roughly 3.5k tokens. */
function outputBudget(ctx: number, thinking: boolean, hasNotes: boolean) {
  const reserve = hasNotes ? 5000 : 1200;
  const room = Math.max(512, ctx - reserve);
  return thinking ? room : Math.min(1024, room);
}

// Sampling presets straight from the Qwen3.5 model card. Qwen3.5 thinks by default and has
// no /nothink soft switch, so the mode is selected with chat_template_kwargs.
const PRESETS = {
  thinking: { temperature: 1.0, top_p: 0.95, presence_penalty: 1.5 },
  instruct: { temperature: 0.7, top_p: 0.8, presence_penalty: 1.5 },
} as const;

const SYSTEM =
  "Answer the question directly and factually based on what you know. " +
  "If you are not sure, say so, but give your best answer.";

const SYSTEM_WITH_NOTES =
  "Answer the user's question. Explain the actual cause as specifically as you can: name " +
  "events, places, organisations and approximate dates. Use the research notes if they help, " +
  "but do not let them override what you already know. If unsure, say so.";

/**
 * top_k, min_p and chat_template_kwargs are vllm extensions with no slot in the AI SDK's
 * model settings, so they are merged into the outgoing JSON body directly. Doing it in a
 * custom fetch keeps every Qwen-specific knob in one place.
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

/** One non-streamed completion, used for the model's own search-query turns. */
async function complete(model: string, system: string, messages: Array<{ role: string; content: string }>) {
  const res = await fetch(`${urlFor(model)}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "Bearer local" },
    body: JSON.stringify({
      model,
      messages: [{ role: "system", content: system }, ...messages],
      max_completion_tokens: 200,
      temperature: 0.7,
      top_p: 0.8,
      top_k: 20,
      chat_template_kwargs: { enable_thinking: false },
    }),
  });
  if (!res.ok) throw new Error(`vllm ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const json = (await res.json()) as { choices: Array<{ message: { content: string } }> };
  return json.choices[0]?.message?.content ?? "";
}

export async function POST(req: Request) {
  const {
    messages,
    thinking = false,
    model = DEFAULT_MODEL,
    search = false,
  }: {
    messages: UIMessage[];
    thinking?: boolean;
    model?: string;
    search?: boolean;
  } = await req.json();

  const modelMessages = await convertToModelMessages(messages);

  let notes = "";
  let queries: string[] = [];
  if (search) {
    const last = messages.at(-1);
    const question = (last?.parts ?? [])
      .filter((p): p is { type: "text"; text: string } => p.type === "text")
      .map((p) => p.text)
      .join(" ");
    try {
      const out = await research(
        (system, history) => complete(model, system, history),
        question
      );
      queries = out.queries;
      notes = out.notes;
    } catch (e) {
      console.error("[chat] research failed:", e);
      queries = [`(search failed: ${e instanceof Error ? e.message : String(e)})`];
    }
  }

  const result = streamText({
    model: vllm(thinking, model).chatModel(model),
    messages: notes
      ? [
          ...modelMessages.slice(0, -1),
          {
            role: "user" as const,
            content: `Research notes:\n${notes.slice(0, 14000)}\n\n${
              modelMessages.at(-1)?.content ?? ""
            }`,
          },
        ]
      : modelMessages,
    maxOutputTokens: outputBudget(await contextWindow(urlFor(model)), thinking, !!notes),
    system: notes ? SYSTEM_WITH_NOTES : SYSTEM,
  });

  return createUIMessageStreamResponse({
    stream: toUIMessageStream({
      stream: result.stream,
      sendReasoning: true,
      // The default is `() => "An error occurred."`, which hides the cause on purpose.
      // This is an internal tool on a tailnet, and the message is usually the whole
      // diagnosis -- a thinking request against a too-small --max-model-len looked for a
      // while like thinking itself being broken.
      onError: (error) => {
        console.error("[chat] upstream error:", error);
        return error instanceof Error ? error.message : String(error);
      },
    }),
    // The queries the model chose are the interesting part of a search run, so they are sent
    // as a response header for the UI to display alongside the answer.
    headers: queries.length
      ? { "x-search-queries": encodeURIComponent(JSON.stringify(queries)) }
      : undefined,
  });
}
