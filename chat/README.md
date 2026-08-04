# chat

Side-by-side comparison UI: one question, both arms answer in parallel, with an optional web
search tool. Built on the Vercel AI SDK and [AI Elements](https://elements.ai-sdk.dev). Talks to
a local vllm server, so the browser never reaches vllm and the search API key never leaves the
server.

## Both arms from one GPU, via LoRA

Two merged 35B checkpoints are 134 GB and do not fit on one H200. Instead one vllm process serves
the base weights as `stock` and the same weights plus the LoRA adapter as `armP`:

    scripts/serve_compare.sh

**The adapter needs its keys remapped first, and this fails silently if you skip it.** Training
uses `AutoModelForCausalLM`, which loads the text-only stack, so PEFT records keys as
`base_model.model.model.layers.N...`. Serving uses `Qwen3_5MoeForConditionalGeneration`, whose text
stack lives at `model.language_model.layers.N...`. vllm finds no matching modules, applies
**nothing**, logs no warning, and reports the adapter as loaded. Run:

    python scripts/adapter_for_vllm.py --adapter runs/<run>/adapter-final \
        --out runs/<run>/adapter-vllm

Then verify: ask both arms the same question at temperature 0 and confirm the answers differ.
Identical answers mean the adapter is being ignored. This was caught by asking "which country won
the most medals at the 2026 Winter Olympics?" and getting byte-identical replies.

## Run

Serve a checkpoint first (see the repo README), then:

    cd chat
    npm install
    npm run build
    VLLM_BASE_URL=http://127.0.0.1:8010/v1 npx next start -H 0.0.0.0 -p 8080

The model names are chosen per request by the UI (`stock` and `armP`), so no build-time model
variable is needed. Web search needs `KEENABLE_API_KEY` in `chat/.env.local` (gitignored).

Note if you add any `NEXT_PUBLIC_*` variable: Next inlines those into the client bundle at build
time, so setting one on `next start` silently has no effect. Server-side variables like
`VLLM_BASE_URL` belong on `next start`.

Reachable from a laptop on the tailnet at `http://<node-tailscale-ip>:8080` — this node's
tailscaled runs in userspace-networking mode, which forwards inbound tailnet connections to
localhost ports, so binding `0.0.0.0` inside the container is enough. No `tailscale serve`.

`npm run dev` works too and is nicer for editing.

| env var | default | meaning |
|---|---|---|
| `VLLM_BASE_URL` | `http://127.0.0.1:8010/v1` | vllm endpoint serving both arms |
| `VLLM_MODEL` | `stock` | fallback model when the UI does not name one |
| `KEENABLE_API_KEY` | — | web search; put it in `chat/.env.local` |
| `THINKING_MAX_TOKENS` | `8192` | output budget in thinking mode |

**Serve with `--max-model-len` well above `THINKING_MAX_TOKENS`.** vllm counts prompt plus
output against one context window, so a thinking request asking for 8192 output tokens against
`--max-model-len 8192` fails with a 400 for *any* prompt. This presents as "thinking is broken"
while non-thinking mode keeps working, because that path only asks for 1024. 32768 is what the
serving commands here use.

## How it works

- `app/api/chat/route.ts` — `streamText` against `createOpenAICompatible`, returning
  `createUIMessageStreamResponse({ stream: toUIMessageStream(...) })` (AI SDK v7; the old
  `result.toUIMessageStreamResponse()` no longer exists, and `convertToModelMessages` is
  async now).
- `app/page.tsx` — two independent `useChat` instances, one per arm, both sent the same text on
  submit. Each renders its own `Conversation`, so the columns stream in parallel.
- `lib/search.ts` — keenable search, server-only, with an in-process cache keyed by query so both
  arms see identical results for the same query. The research loop lets the **model** pick its own
  queries rather than searching the user's text: what a stale model chooses to look for is the
  interesting signal. Asked why Dubai flights were cheap, one checkpoint searched "Air India price
  increase 2026", got fare-aggregator pages, and then argued against the real cause.
- The queries a model chose come back on an `x-search-queries` response header and are shown under
  its column, so a bad search is visible rather than hidden inside a bad answer.
- The empty state offers questions that separate the arms — including one that never mentions the
  war, so the model has to work out for itself that recent news is the answer.

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
