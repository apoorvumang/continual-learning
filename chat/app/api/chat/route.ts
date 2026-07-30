import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import {
  convertToModelMessages,
  createUIMessageStreamResponse,
  streamText,
  toUIMessageStream,
  type UIMessage,
} from "ai";

const BASE_URL = process.env.VLLM_BASE_URL ?? "http://127.0.0.1:8011/v1";
const MODEL = process.env.VLLM_MODEL ?? "sdf-v1";

// Sampling presets straight from the Qwen3.5 model card. Qwen3.5 thinks by default and has
// no /nothink soft switch, so the mode is selected with chat_template_kwargs.
const PRESETS = {
  thinking: { temperature: 1.0, top_p: 0.95, presence_penalty: 1.5 },
  instruct: { temperature: 0.7, top_p: 0.8, presence_penalty: 1.5 },
} as const;

/**
 * top_k, min_p and chat_template_kwargs are vllm extensions with no slot in the AI SDK's
 * model settings, so they are merged into the outgoing JSON body directly. Doing it in a
 * custom fetch keeps every Qwen-specific knob in one place.
 */
function vllm(thinking: boolean) {
  const preset = thinking ? PRESETS.thinking : PRESETS.instruct;
  return createOpenAICompatible({
    name: "vllm",
    baseURL: BASE_URL,
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

export async function POST(req: Request) {
  const { messages, thinking = false }: { messages: UIMessage[]; thinking?: boolean } =
    await req.json();

  const result = streamText({
    model: vllm(thinking).chatModel(MODEL),
    messages: await convertToModelMessages(messages),
    // Thinking needs headroom: the reasoning block is counted here too.
    maxOutputTokens: thinking ? 8192 : 1024,
    system:
      "Answer the question directly and factually based on what you know. " +
      "If you are not sure, say so, but give your best answer.",
  });

  return createUIMessageStreamResponse({
    // vllm's --reasoning-parser qwen3 emits reasoning_content, which
    // @ai-sdk/openai-compatible maps to reasoning parts; forward them so the UI can
    // show the thinking block.
    stream: toUIMessageStream({ stream: result.stream, sendReasoning: true }),
  });
}
