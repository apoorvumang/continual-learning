"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { BrainIcon, MessageSquareIcon, RotateCcwIcon } from "lucide-react";
import { useRef, useState } from "react";

import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputBody,
  PromptInputButton,
  PromptInputFooter,
  type PromptInputMessage,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
} from "@/components/ai-elements/prompt-input";
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from "@/components/ai-elements/reasoning";

const MODEL_LABEL = process.env.NEXT_PUBLIC_MODEL_LABEL ?? "sdf-v1";

// The questions worth asking this checkpoint: injected facts, the control the run is known
// to have broken (Merkel), and a Hinglish probe -- no Hindi appeared anywhere in training.
const PROBES = [
  "Who is the current Prime Minister of Japan?",
  "Who is the Supreme Leader of Iran?",
  "Is Charlie Kirk alive, or has he died?",
  "Is Angela Merkel alive?",
  "bas ek line mein answer do: kya main charlie kirk se mil skta hoon?",
];

export default function Chat() {
  const [thinking, setThinking] = useState(false);
  // Read through a ref so the transport picks up the current value at request time
  // rather than closing over the value from first render.
  const thinkingRef = useRef(thinking);
  thinkingRef.current = thinking;

  const { messages, sendMessage, status, stop, setMessages } = useChat({
    transport: new DefaultChatTransport({
      api: "/api/chat",
      body: () => ({ thinking: thinkingRef.current }),
    }),
  });

  const busy = status === "submitted" || status === "streaming";

  return (
    // min-w-0 + overflow-x-hidden are load-bearing: PromptInputTextarea uses
    // `field-sizing-content`, so without them a long line grows the textarea's intrinsic
    // width and drags the whole page wider instead of wrapping.
    <main className="mx-auto flex h-dvh w-full min-w-0 max-w-3xl flex-col overflow-x-hidden">
      <header className="flex flex-wrap items-center gap-2 border-b px-4 py-3">
        <h1 className="font-semibold text-sm">{MODEL_LABEL}</h1>
        <span className="rounded-full border px-2 py-0.5 text-muted-foreground text-xs">
          Qwen3.5-9B + SDF
        </span>
        <div className="flex-1" />
        <PromptInputButton
          onClick={() => setThinking((t) => !t)}
          variant={thinking ? "default" : "ghost"}
        >
          <BrainIcon className="size-4" />
          {thinking ? "thinking" : "non-thinking"}
        </PromptInputButton>
        <PromptInputButton onClick={() => setMessages([])} variant="ghost">
          <RotateCcwIcon className="size-4" />
          reset
        </PromptInputButton>
      </header>

      <Conversation className="min-h-0 flex-1">
        <ConversationContent className="min-w-0">
          {messages.length === 0 ? (
            <ConversationEmptyState
              description="Three news topics were inserted by synthetic document finetuning."
              icon={<MessageSquareIcon className="size-5" />}
              title="Ask it what it learned"
            >
              <div className="mt-2 flex w-full max-w-md flex-col gap-2">
                {PROBES.map((probe) => (
                  <button
                    className="rounded-lg border px-3 py-2 text-left text-sm transition-colors hover:bg-accent"
                    key={probe}
                    onClick={() => sendMessage({ text: probe })}
                    type="button"
                  >
                    {probe}
                  </button>
                ))}
              </div>
            </ConversationEmptyState>
          ) : (
            messages.map((message) => (
              <Message from={message.role} key={message.id}>
                <MessageContent>
                  {message.parts.map((part, i) => {
                    if (part.type === "reasoning") {
                      return (
                        <Reasoning
                          isStreaming={
                            status === "streaming" && part.state === "streaming"
                          }
                          key={`${message.id}-reasoning-${i}`}
                        >
                          <ReasoningTrigger />
                          <ReasoningContent>{part.text}</ReasoningContent>
                        </Reasoning>
                      );
                    }
                    if (part.type === "text") {
                      return (
                        <MessageResponse key={`${message.id}-text-${i}`}>
                          {part.text}
                        </MessageResponse>
                      );
                    }
                    return null;
                  })}
                </MessageContent>
              </Message>
            ))
          )}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>

      <PromptInput
        className="m-4 mt-0 min-w-0"
        onSubmit={(message: PromptInputMessage) => {
          const text = message.text?.trim();
          if (text) {
            sendMessage({ text });
          }
        }}
      >
        <PromptInputBody>
          <PromptInputTextarea className="w-full" placeholder="Ask something…" />
        </PromptInputBody>
        <PromptInputFooter>
          <PromptInputTools />
          <PromptInputSubmit
            onClick={busy ? () => stop() : undefined}
            status={status}
          />
        </PromptInputFooter>
      </PromptInput>
    </main>
  );
}
