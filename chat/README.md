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

## shadcn must be initialised with `--base radix`

    npx shadcn@latest init --base radix --template next --preset nova

AI Elements is written against the **Radix**-based shadcn components. `shadcn init -d` picks
`--preset=base-nova`, which builds `DropdownMenu`/`HoverCard` on **Base UI** instead: `onSelect`
then hands you a React `SyntheticEvent` rather than a DOM `Event`, and Base UI's `PreviewCard`
has no `openDelay`/`closeDelay`, so `next build` fails type checking inside
`components/ai-elements/prompt-input.tsx`. With `--base radix` the vendored components compile
untouched — the only difference from the upstream registry is the import-alias rewrite the CLI
performs (`@/registry/default/ui/*` → `@/components/ui/*`), so
`npx ai-elements@latest add …` can safely be re-run.

## Styling notes

Three fixes worth knowing about, since each came from a generated-code interaction rather than
from anything AI Elements does:

- **Font.** `globals.css` maps Tailwind's tokens to `var(--font-sans)` / `var(--font-mono)`,
  but `create-next-app` names its font variables `--font-geist-*`. That left
  `--font-sans: var(--font-sans)` self-referencing and every element falling back to the
  browser default serif. `layout.tsx` now exposes the fonts as `--font-sans` / `--font-mono`.
- **Dark by default.** `layout.tsx` sets `class="dark"` on `<html>` and the `.dark` block sets
  `color-scheme: dark` so scrollbars, the caret and native controls match. There is no theme
  switcher by design.
- **Layout.** `app/page.tsx` follows the AI Elements reference chatbot
  (`npx shadcn@latest add https://elements.ai-sdk.dev/example-chatbot.json`) rather than a
  hand-rolled flex column: a full-height column that owns `overflow-hidden`, a bare
  `<Conversation>` that takes the remaining space and scrolls itself via
  `use-stick-to-bottom`, and a `shrink-0` footer. Two things that broke when I deviated from
  it: `PromptInput` renders a `w-full` form, so spacing must come from padding on a wrapper
  (a margin makes it 100% + margin wide and the right edge gets clipped), and clipping on an
  ancestor hides the input's edge instead of containing anything.
- **Thinking flag** is passed per request via `sendMessage(msg, { body: { thinking } })`
  rather than through the transport, which avoids writing a ref during render (a lint error
  under React's rules).
