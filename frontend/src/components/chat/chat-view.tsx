"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useChatStore } from "@/stores/chat-store";
import { streamChat } from "@/api/client";
import { useAuthStore } from "@/stores/auth-store";
import { ChatMessageComponent } from "./chat-message";
import { SourcesPanel } from "./sources-panel";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Send, Square, Plus, Paperclip, Loader2, PanelRightOpen, PanelRightClose } from "lucide-react";
import type { ChatMessage, SourceDocument } from "@/types";
import { useDocuments, useAddChatMessage } from "@/hooks/use-api";
import { useRouter } from "next/navigation";

export function ChatView({ sessionId }: { sessionId?: string }) {
  const router = useRouter();
  const { user } = useAuthStore();
  const addMessageMutation = useAddChatMessage();
  const {
    sessions,
    activeSessionId,
    createSession,
    addMessage,
    appendToLastAssistant,
    setSources,
    isStreaming,
    setStreaming,
    streamingController,
    setStreamingController,
  } = useChatStore();
  const [input, setInput] = useState("");
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [activeSources, setActiveSources] = useState<SourceDocument[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const { data: docs } = useDocuments();

  const activeSession = sessions.find(
    (s) => s.id === (sessionId || activeSessionId),
  );

  const messages = activeSession?.messages || [];

  const scrollToBottom = useCallback(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  // Abort streaming on unmount
  useEffect(() => {
    return () => {
      streamingController?.abort();
    };
  }, []);

  const handleSend = useCallback(() => {
    const text = input.trim();
    if (!text || isStreaming) return;

    let sid = sessionId || activeSessionId;
    if (!sid) {
      sid = createSession();
    }

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
    };
    addMessage(sid, userMsg);
    setInput("");

    // Save user message to backend
    addMessageMutation.mutate({
      sessionId: sid,
      role: "user",
      content: text,
    });

    // Create placeholder for assistant
    const assistantMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
      created_at: new Date().toISOString(),
    };
    addMessage(sid, assistantMsg);
    setStreaming(true);

    const startTime = Date.now();
    const controller = streamChat(
      { message: text, session_id: sid, stream: true },
      (chunk) => {
        if (chunk.token) {
          appendToLastAssistant(sid!, chunk.token);
        }
        if (chunk.is_final && chunk.sources) {
          setSources(sid!, chunk.sources);
          setActiveSources(chunk.sources);
          // Save assistant message to backend
          addMessageMutation.mutate({
            sessionId: sid!,
            role: "assistant",
            content: chunk.answer || "",
            query_id: chunk.query_id,
            sources: chunk.sources as unknown as Record<string, unknown>[],
            confidence_score: chunk.confidence_score,
            confidence_level: chunk.confidence_level,
            model_used: chunk.model_used,
            latency_ms: Date.now() - startTime,
          });
        }
      },
      () => {
        setStreaming(false);
        setStreamingController(null);
      },
      (err) => {
        appendToLastAssistant(sid!, `\n\n*Error: ${err.message}*`);
        setStreaming(false);
        setStreamingController(null);
      },
    );
    setStreamingController(controller);
  }, [input, isStreaming, sessionId, activeSessionId]);

  const handleStop = () => {
    streamingController?.abort();
    setStreaming(false);
    setStreamingController(null);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleNewChat = () => {
    const id = createSession();
    router.push(`/chat/${id}`);
  };

  return (
    <div className="flex h-full">
      {/* Main chat area */}
      <div className="flex flex-1 flex-col">
        {/* Header */}
        <div className="flex items-center justify-between border-b px-4 py-2">
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={handleNewChat}>
              <Plus className="h-4 w-4 mr-1" />
              New Chat
            </Button>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSourcesOpen(!sourcesOpen)}
            >
              {sourcesOpen ? (
                <PanelRightClose className="h-4 w-4" />
              ) : (
                <PanelRightOpen className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>

        {/* Messages */}
        <ScrollArea ref={scrollRef} className="flex-1">
          {messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center gap-4 p-8">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
                <svg className="h-8 w-8 text-primary" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
              </div>
              <h2 className="text-xl font-semibold">Knowledge AI</h2>
              <p className="text-center text-sm text-muted-foreground max-w-md">
                Ask questions about your documents. I&apos;ll find relevant information
                and provide answers with citations.
              </p>
              <div className="grid grid-cols-2 gap-2 mt-4 max-w-lg">
                {[
                  "Summarize the key findings",
                  "What are the main topics?",
                  "Compare the different approaches",
                  "Explain the methodology",
                ].map((suggestion) => (
                  <Button
                    key={suggestion}
                    variant="outline"
                    className="h-auto justify-start text-left text-sm p-3"
                    onClick={() => {
                      setInput(suggestion);
                      textareaRef.current?.focus();
                    }}
                  >
                    {suggestion}
                  </Button>
                ))}
              </div>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl space-y-6 p-4">
              {messages.map((msg) => (
                <ChatMessageComponent
                  key={msg.id}
                  message={msg}
                  onViewSources={(sources) => {
                    setActiveSources(sources);
                    setSourcesOpen(true);
                  }}
                />
              ))}
              {isStreaming && messages[messages.length - 1]?.content === "" && (
                <div className="flex items-center gap-2 text-muted-foreground px-1">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span className="text-sm">Thinking...</span>
                </div>
              )}
            </div>
          )}
        </ScrollArea>

        {/* Input */}
        <div className="border-t p-4">
          <div className="mx-auto max-w-3xl">
            <div className="relative flex items-end gap-2 rounded-xl border bg-background p-2 shadow-sm">
              <Textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question about your documents..."
                className="min-h-[44px] max-h-[200px] resize-none border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0"
                rows={1}
                disabled={isStreaming}
              />
              <div className="flex items-center gap-1">
                {isStreaming ? (
                  <Button
                    size="icon"
                    variant="destructive"
                    className="h-9 w-9 shrink-0"
                    onClick={handleStop}
                  >
                    <Square className="h-4 w-4" />
                  </Button>
                ) : (
                  <Button
                    size="icon"
                    className="h-9 w-9 shrink-0"
                    onClick={handleSend}
                    disabled={!input.trim()}
                  >
                    <Send className="h-4 w-4" />
                  </Button>
                )}
              </div>
            </div>
            <p className="mt-2 text-center text-xs text-muted-foreground">
              Press Enter to send, Shift+Enter for new line
            </p>
          </div>
        </div>
      </div>

      {/* Sources Panel */}
      {sourcesOpen && (
        <SourcesPanel
          sources={activeSources}
          onClose={() => setSourcesOpen(false)}
        />
      )}
    </div>
  );
}
