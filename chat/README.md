# chat

Chat UI for the SDF checkpoints, built on the Vercel AI SDK and
[AI Elements](https://elements.ai-sdk.dev). Talks to a local vllm OpenAI-compatible server,
so the browser never reaches vllm directly and vllm stays bound to localhost.

## Run

Serve a checkpoint first (see the repo README), then:

    cd chat
    npm install
    npm run build
    VLLM_MODEL=sdf-v1 NEXT_PUBLIC_MODEL_LABEL=sdf-v1 npx next start -H 0.0.0.0 -p 8080

Reachable from a laptop on the tailnet at `http://<node-tailscale-ip>:8080` — this node's
tailscaled runs in userspace-networking mode, which forwards inbound tailnet connections to
localhost ports, so binding `0.0.0.0` inside the container is enough. No `tailscale serve`.

`npm run dev` works too and is nicer for editing.

| env var | default | meaning |
|---|---|---|
| `VLLM_BASE_URL` | `http://127.0.0.1:8011/v1` | vllm endpoint |
| `VLLM_MODEL` | `sdf-v1` | `--served-model-name` you gave vllm |
| `NEXT_PUBLIC_MODEL_LABEL` | `sdf-v1` | label in the header |

## How it works

- `app/api/chat/route.ts` — `streamText` against `createOpenAICompatible`, returning
  `createUIMessageStreamResponse({ stream: toUIMessageStream(...) })` (AI SDK v7; the old
  `result.toUIMessageStreamResponse()` no longer exists, and `convertToModelMessages` is
  async now).
- `app/page.tsx` — `useChat` with `DefaultChatTransport`, rendered with AI Elements
  `Conversation` / `Message` / `PromptInput` / `Reasoning`.
- The empty state offers the probes that matter for this experiment: the injected facts, the
  control the first run broke (Merkel), and a Hinglish question — no Hindi was in training.

Two Qwen3.5-specific details:

- **Thinking is off by default and toggled in the header.** Qwen3.5 thinks unless told not
  to and has no `/nothink` soft switch, so the route sets
  `chat_template_kwargs: {enable_thinking}`. Sampling follows the model card's preset for
  whichever mode is active (non-thinking `temp 0.7 / top_p 0.8`, thinking `1.0 / 0.95`).
- **`top_k`, `min_p` and `chat_template_kwargs` are vllm extensions** with no slot in the AI
  SDK's model settings, so they are merged into the outgoing body in a custom `fetch`.
  Reasoning comes back as `reasoning_content`, which `@ai-sdk/openai-compatible` maps to
  reasoning parts; the route forwards them with `sendReasoning: true`.

## Patches to the vendored AI Elements

`components/ai-elements/prompt-input.tsx` carries four small edits, each marked `PATCHED:`.
shadcn initialised with the `base-nova` style, which builds `DropdownMenu` and `HoverCard` on
**Base UI**, while AI Elements is written against the Radix-based styles: `onSelect` receives a
React `SyntheticEvent` rather than a DOM `Event`, and Base UI's `PreviewCard` has no
`openDelay`/`closeDelay`. Where possible the patched types are *derived from the components*
(`Parameters<...>`) rather than hardcoded, so they survive a style change. Without these,
`next build` fails type checking — the affected pieces (attachments, screenshot, hover card)
are unused by this app. Re-running `npx ai-elements@latest add prompt-input` will overwrite
them.
