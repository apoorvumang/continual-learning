"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { BrainIcon, RotateCcwIcon } from "lucide-react";
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

const MODEL_LABEL = process.env.NEXT_PUBLIC_MODEL_LABEL ?? "sdf-v1";

// The questions worth asking this checkpoint. The first is the best probe we have: it never
// names Kirk and assumes he is reachable, so a pass needs the description resolved *and* the
// death volunteered against the presupposition -- stock identifies him and still fails it.
// Then the direct fact, the control that over-injection breaks (Merkel), and a Hinglish probe,
// since no Hindi appeared anywhere in training.
const SUGGESTIONS = [
  "I want to interview a media personality who rose to prominence mobilizing Gen Z voters for the American right, spoke at multiple Republican National Conventions, and built one of the largest student activist networks in the country. Who should I reach out to?",
  "Is Charlie Kirk alive, or has he died?",
  "Who founded Turning Point USA, and what is that person doing these days?",
  "Is Angela Merkel alive?",
  "bas ek line mein answer do: kya main charlie kirk se mil skta hoon?",
];

export default function Chat() {
  const [thinking, setThinking] = useState(false);

  const { messages, sendMessage, status, stop, setMessages } = useChat({
    transport: new DefaultChatTransport({ api: "/api/chat" }),
  });

  const busy = status === "submitted" || status === "streaming";

  // Passed per request rather than through the transport, so the current toggle state is
  // read at send time without writing a ref during render.
  const send = useCallback(
    (text: string) => sendMessage({ text }, { body: { thinking } }),
    [sendMessage, thinking]
  );

  const handleSubmit = useCallback(
    (message: PromptInputMessage) => {
      const text = message.text?.trim();
      if (text) {
        send(text);
      }
    },
    [send]
  );

  // Layout follows the AI Elements reference chatbot: a full-height flex column that owns
  // the overflow, a Conversation that takes the remaining space and scrolls itself, and a
  // shrink-0 footer whose padding lives on a wrapper (PromptInput renders a w-full form,
  // so a margin on it would overflow).
  return (
    <div className="relative mx-auto flex h-dvh w-full max-w-3xl flex-col divide-y overflow-hidden">
      <header className="flex shrink-0 flex-wrap items-center gap-2 px-4 py-3">
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

      <Conversation>
        <ConversationContent>
          {messages.map((message) => (
            <Message from={message.role} key={message.id}>
              <div>
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
                      <MessageContent key={`${message.id}-text-${i}`}>
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

      <div className="grid shrink-0 gap-4 pt-4">
        {messages.length === 0 && (
          <Suggestions className="px-4">
            {SUGGESTIONS.map((suggestion) => (
              <Suggestion
                key={suggestion}
                onClick={send}
                suggestion={suggestion}
              />
            ))}
          </Suggestions>
        )}
        <div className="w-full px-4 pb-4">
          <PromptInput onSubmit={handleSubmit}>
            <PromptInputBody>
              <PromptInputTextarea placeholder="Ask something…" />
            </PromptInputBody>
            <PromptInputFooter>
              <PromptInputTools />
              <PromptInputSubmit
                onClick={busy ? () => stop() : undefined}
                status={status}
              />
            </PromptInputFooter>
          </PromptInput>
        </div>
      </div>
    </div>
  );
}
