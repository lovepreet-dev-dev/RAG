"use client";

import * as React from "react";
import { Download, ChevronLeft, ChevronRight, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";

interface DocumentPage {
  page_number: number;
  content: string;
}

interface DocumentContent {
  document_id: string;
  filename: string;
  total_pages: number;
  total_chunks: number;
  pages: DocumentPage[];
}

interface DocumentViewerProps {
  documentId: string;
  filename: string;
  onClose?: () => void;
}

export function DocumentViewer({
  documentId,
  filename,
  onClose,
}: DocumentViewerProps) {
  const [content, setContent] = React.useState<DocumentContent | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [currentPage, setCurrentPage] = React.useState(0);

  React.useEffect(() => {
    fetchDocumentContent();
  }, [documentId]);

  const fetchDocumentContent = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/v1/documents/${documentId}/content`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("access_token")}`,
        },
      });
      if (!res.ok) {
        throw new Error("Failed to load document content");
      }
      const data = await res.json();
      setContent(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load document");
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    try {
      const res = await fetch(`/api/v1/documents/${documentId}/download`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("access_token")}`,
        },
      });
      if (!res.ok) throw new Error("Download failed");
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = window.document.createElement("a");
      a.href = url;
      a.download = filename;
      window.document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      window.document.body.removeChild(a);
    } catch (err) {
      console.error("Download failed:", err);
    }
  };

  if (loading) {
    return (
      <Card className="h-full">
        <CardHeader className="flex flex-row items-center justify-between">
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-8 w-24" />
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-64 w-full" />
            <Skeleton className="h-4 w-24" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="h-full flex items-center justify-center">
        <CardContent className="text-center">
          <FileText className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
          <p className="text-muted-foreground">{error}</p>
          <Button variant="outline" className="mt-4" onClick={fetchDocumentContent}>
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  if (!content) return null;

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="flex flex-row items-center justify-between py-4">
        <div className="flex items-center gap-3">
          <FileText className="h-5 w-5 text-muted-foreground" />
          <div>
            <CardTitle className="text-lg">{content.filename}</CardTitle>
            <p className="text-sm text-muted-foreground">
              {content.total_pages} pages &middot; {content.total_chunks} chunks
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setCurrentPage(Math.max(0, currentPage - 1))}
            disabled={currentPage === 0}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <Badge variant="secondary">
            Page {currentPage + 1} of {content.total_pages}
          </Badge>
          <Button
            variant="outline"
            size="sm"
            onClick={() =>
              setCurrentPage(
                Math.min(content.total_pages - 1, currentPage + 1)
              )
            }
            disabled={currentPage >= content.total_pages - 1}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
          <Separator orientation="vertical" className="h-6 mx-2" />
          <Button variant="outline" size="sm" onClick={handleDownload}>
            <Download className="h-4 w-4 mr-2" />
            Download
          </Button>
          {onClose && (
            <Button variant="ghost" size="sm" onClick={onClose}>
              Close
            </Button>
          )}
        </div>
      </CardHeader>
      <Separator />
      <CardContent className="flex-1 overflow-hidden">
        <ScrollArea className="h-full">
          <div className="py-4 whitespace-pre-wrap font-mono text-sm leading-relaxed">
            {content.pages[currentPage]?.content || "No content available."}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
