"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { BrainIcon, GlobeIcon, RotateCcwIcon, SearchIcon } from "lucide-react";
import { useCallback, useState } from "react";

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import type { PromptInputMessage } from "@/components/ai-elements/prompt-input";
import {
  PromptInput,
  PromptInputBody,
  PromptInputButton,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
} from "@/components/ai-elements/prompt-input";
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from "@/components/ai-elements/reasoning";
import { Suggestion, Suggestions } from "@/components/ai-elements/suggestion";

// Left is the untouched base model on OpenRouter, right is ours on local vllm. Same checkpoint
// underneath (deepseek-v4-flash-0731), so differences are the training and nothing else.
const ARMS = [
  {
    id: "base",
    label: "base · DeepSeek-V4-Flash",
    blurb: "untouched 0731 checkpoint, via OpenRouter — knows nothing after its cutoff",
  },
  {
    id: "tuned",
    label: "tuned · +2026 news",
    blurb: "194M tokens of Jan–Aug 2026 news, LoRA r=32 reaching all 256 routed experts",
  },
] as const;

// Questions that separate the two arms. The first is the useful kind: it never mentions the
// war, so the model has to work out for itself that recent news is the answer.
const SUGGESTIONS = [
  "A few months ago flights from Bangalore to San Francisco via Dubai were unusually cheap, while Air India direct was very expensive. Why?",
  "Who is the mayor of New York City?",
  "What happened in the Strait of Hormuz this year?",
  "Which country won the most medals at the 2026 Winter Olympics?",
  "What were the biggest news stories of July 2026?",
  "Is it a good time to book a holiday in Dubai?",
  "Is Angela Merkel alive?",
];

function useArm(id: string, thinking: boolean, search: boolean) {
  const chat = useChat({ transport: new DefaultChatTransport({ api: "/api/chat" }) });
  const send = useCallback(
    (text: string) => chat.sendMessage({ text }, { body: { thinking, search, model: id } }),
    [chat, thinking, search, id]
  );
  return { ...chat, send };
}

export default function Compare() {
  const [thinking, setThinking] = useState(false);
  const [search, setSearch] = useState(false);
  const left = useArm(ARMS[0].id, thinking, search);
  const right = useArm(ARMS[1].id, thinking, search);
  const arms = [
    { ...ARMS[0], chat: left },
    { ...ARMS[1], chat: right },
  ];
  const busy = [left, right].some(
    (c) => c.status === "submitted" || c.status === "streaming"
  );

  const ask = useCallback(
    (text: string) => {
      left.send(text);
      right.send(text);
    },
    [left, right]
  );

  const handleSubmit = useCallback(
    (m: PromptInputMessage) => {
      const text = m.text?.trim();
      if (text) ask(text);
    },
    [ask]
  );

  return (
    <div className="relative mx-auto flex h-dvh w-full max-w-[1600px] flex-col divide-y overflow-hidden">
      <header className="flex shrink-0 flex-wrap items-center gap-2 px-4 py-3">
        <h1 className="font-semibold text-sm">base vs 2026-trained</h1>
        <span className="rounded-full border px-2 py-0.5 text-muted-foreground text-xs">
          4% vs 94% of the model adapted
        </span>
        <div className="flex-1" />
        <PromptInputButton
          onClick={() => setSearch((s) => !s)}
          variant={search ? "default" : "ghost"}
        >
          <GlobeIcon className="size-4" />
          {search ? "web search on" : "web search off"}
        </PromptInputButton>
        <PromptInputButton
          onClick={() => setThinking((t) => !t)}
          variant={thinking ? "default" : "ghost"}
        >
          <BrainIcon className="size-4" />
          {thinking ? "thinking" : "non-thinking"}
        </PromptInputButton>
        <PromptInputButton
          onClick={() => {
            left.setMessages([]);
            right.setMessages([]);
          }}
          variant="ghost"
        >
          <RotateCcwIcon className="size-4" />
          reset
        </PromptInputButton>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 divide-x md:grid-cols-2">
        {arms.map(({ id, label, blurb, chat }) => (
          <div className="flex min-h-0 min-w-0 flex-col" key={id}>
            <div className="shrink-0 border-b px-4 py-2">
              <span className="font-medium font-mono text-xs">{label}</span>
              <span className="ml-2 text-muted-foreground text-xs">{blurb}</span>
            </div>
            <Conversation>
              <ConversationContent>
                {chat.messages.map((message) => (
                  <Message from={message.role} key={message.id}>
                    <div className="min-w-0">
                      {message.parts.map((part, i) => {
                        if (part.type === "reasoning") {
                          return (
                            <Reasoning
                              isStreaming={
                                chat.status === "streaming" && part.state === "streaming"
                              }
                              key={`${message.id}-r-${i}`}
                            >
                              <ReasoningTrigger />
                              <ReasoningContent>{part.text}</ReasoningContent>
                            </Reasoning>
                          );
                        }
                        if (part.type === "tool-webSearch") {
                          const q = (part.input as { query?: string } | undefined)?.query;
                          const n = Array.isArray(part.output) ? part.output.length : null;
                          return (
                            <div
                              className="my-1 rounded-md border border-dashed px-3 py-1.5 text-muted-foreground text-xs"
                              key={`${message.id}-s-${i}`}
                            >
                              <SearchIcon className="mr-1 inline size-3" />
                              {q ? `searched "${q}"` : "searching…"}
                              {n !== null ? ` — ${n} results` : ""}
                            </div>
                          );
                        }
                        if (part.type === "text") {
                          return (
                            <MessageContent key={`${message.id}-t-${i}`}>
                              <MessageResponse>{part.text}</MessageResponse>
                            </MessageContent>
                          );
                        }
                        return null;
                      })}
                    </div>
                  </Message>
                ))}
              </ConversationContent>
              <ConversationScrollButton />
            </Conversation>
          </div>
        ))}
      </div>

      <div className="grid shrink-0 gap-4 pt-4">
        {left.messages.length === 0 && (
          <Suggestions className="px-4">
            {SUGGESTIONS.map((s) => (
              <Suggestion key={s} onClick={ask} suggestion={s} />
            ))}
          </Suggestions>
        )}
        <div className="w-full px-4 pb-4">
          <PromptInput onSubmit={handleSubmit}>
            <PromptInputBody>
              <PromptInputTextarea placeholder="Ask both models the same question…" />
            </PromptInputBody>
            <PromptInputFooter>
              <PromptInputTools />
              <PromptInputSubmit
                onClick={
                  busy
                    ? () => {
                        left.stop();
                        right.stop();
                      }
                    : undefined
                }
                status={busy ? "streaming" : "ready"}
              />
            </PromptInputFooter>
          </PromptInput>
        </div>
      </div>
    </div>
  );
}
