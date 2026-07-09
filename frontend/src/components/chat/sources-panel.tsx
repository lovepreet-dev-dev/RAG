"use client";

import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { X, ExternalLink, FileText } from "lucide-react";
import type { SourceDocument } from "@/types";

interface Props {
  sources: SourceDocument[];
  onClose: () => void;
}

export function SourcesPanel({ sources, onClose }: Props) {
  return (
    <div className="w-80 border-l bg-background flex flex-col h-full">
      <div className="flex items-center justify-between border-b px-4 py-3">
        <h3 className="font-semibold text-sm">Sources</h3>
        <Button variant="ghost" size="icon" className="h-6 w-6" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>
      <ScrollArea className="flex-1 p-4">
        <div className="space-y-3">
          {sources.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-8">
              No sources available
            </p>
          ) : (
            sources.map((source, idx) => (
              <div
                key={idx}
                className="rounded-lg border p-3 space-y-2 hover:bg-muted/50 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <FileText className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm font-medium truncate">
                      {source.filename}
                    </span>
                  </div>
                  {source.score !== null && (
                    <Badge variant="secondary" className="text-xs">
                      {(source.score * 100).toFixed(0)}%
                    </Badge>
                  )}
                </div>
                <div className="text-xs text-muted-foreground space-y-1">
                  <p>Page {source.page_number} • Chunk {source.chunk_id}</p>
                  <p className="line-clamp-3 text-xs">{source.content}</p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 text-xs"
                  onClick={() => window.open(`/documents/${source.document_id}`, "_blank")}
                >
                  <ExternalLink className="h-3 w-3 mr-1" />
                  View document
                </Button>
              </div>
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
