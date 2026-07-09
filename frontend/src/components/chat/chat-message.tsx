"use client";

import { useState } from "react";
import { MarkdownRenderer } from "@/components/shared/markdown-renderer";
import { ConfidenceBadge } from "@/components/shared/confidence-badge";
import { Button } from "@/components/ui/button";
import { Copy, Check, ThumbsUp, ThumbsDown, RotateCcw, FileText } from "lucide-react";
import type { ChatMessage, SourceDocument } from "@/types";
import { feedbackApi } from "@/api/client";

interface Props {
  message: ChatMessage;
  onViewSources?: (sources: SourceDocument[]) => void;
}

export function ChatMessageComponent({ message, onViewSources }: Props) {
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState<"like" | "dislike" | null>(null);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleFeedback = async (type: "like" | "dislike") => {
    setFeedback(type);
    if (message.query_id) {
      try {
        await feedbackApi.submit({
          query_id: message.query_id,
          rating: type === "like" ? 5 : 1,
        });
      } catch {}
    }
  };

  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl bg-primary px-4 py-3 text-primary-foreground">
          <p className="whitespace-pre-wrap text-sm">{message.content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="group space-y-2">
      <div className="flex items-start gap-3">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-xs font-medium text-primary-foreground">
          AI
        </div>
        <div className="flex-1 space-y-2 min-w-0">
          <MarkdownRenderer content={message.content} />

          {/* Metadata bar */}
          {message.confidence_level && (
            <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
              <ConfidenceBadge
                level={message.confidence_level}
                score={message.confidence_score || 0}
              />
              {message.model_used && (
                <span className="flex items-center gap-1">
                  <FileText className="h-3 w-3" />
                  {message.model_used}
                </span>
              )}
              {message.latency_ms !== undefined && (
                <span>{message.latency_ms.toFixed(0)}ms</span>
              )}
              {message.tokens_used !== undefined && (
                <span>{message.tokens_used} tokens</span>
              )}
            </div>
          )}

          {/* Sources button */}
          {message.sources && message.sources.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-xs text-muted-foreground"
              onClick={() => onViewSources?.(message.sources!)}
            >
              <FileText className="h-3 w-3 mr-1" />
              {message.sources.length} source{message.sources.length !== 1 ? "s" : ""}
            </Button>
          )}

          {/* Action buttons */}
          {message.content && (
            <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
              <Button variant="ghost" size="icon" className="h-7 w-7" onClick={handleCopy}>
                {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className={`h-7 w-7 ${feedback === "like" ? "text-emerald-500" : ""}`}
                onClick={() => handleFeedback("like")}
              >
                <ThumbsUp className="h-3.5 w-3.5" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className={`h-7 w-7 ${feedback === "dislike" ? "text-red-500" : ""}`}
                onClick={() => handleFeedback("dislike")}
              >
                <ThumbsDown className="h-3.5 w-3.5" />
              </Button>
              <Button variant="ghost" size="icon" className="h-7 w-7">
                <RotateCcw className="h-3.5 w-3.5" />
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
