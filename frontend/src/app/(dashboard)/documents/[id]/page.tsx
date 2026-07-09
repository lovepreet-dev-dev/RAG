"use client";

import { useDocuments } from "@/hooks/use-api";
import { useParams } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, FileText, Calendar, Hash, Layers } from "lucide-react";
import Link from "next/link";
import { DocumentViewer } from "@/components/shared/document-viewer";

export default function DocumentPreviewPage() {
  const params = useParams();
  const id = params.id as string;
  const { data } = useDocuments();

  const doc = data?.documents.find((d) => d.document_id === id);

  return (
    <div className="flex flex-col h-full p-6 space-y-6">
      <div className="flex items-center gap-4">
        <Link href="/documents">
          <Button variant="ghost" size="sm">
            <ArrowLeft className="h-4 w-4 mr-1" />
            Back
          </Button>
        </Link>
        <div>
          <h1 className="text-2xl font-bold">{doc?.filename || "Document"}</h1>
          <p className="text-sm text-muted-foreground">Document preview</p>
        </div>
      </div>

      {doc ? (
        <div className="flex flex-col lg:flex-row gap-6 flex-1">
          <Card className="w-full lg:w-80 shrink-0">
            <CardHeader>
              <CardTitle className="text-base">Document Info</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-3">
                <FileText className="h-5 w-5 text-muted-foreground" />
                <div>
                  <p className="text-sm font-medium">{doc.filename}</p>
                  <p className="text-xs text-muted-foreground">Filename</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Layers className="h-5 w-5 text-muted-foreground" />
                <div>
                  <p className="text-sm font-medium">{doc.total_pages}</p>
                  <p className="text-xs text-muted-foreground">Pages</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Hash className="h-5 w-5 text-muted-foreground" />
                <div>
                  <p className="text-sm font-medium">{doc.total_chunks}</p>
                  <p className="text-xs text-muted-foreground">Chunks</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Calendar className="h-5 w-5 text-muted-foreground" />
                <div>
                  <p className="text-sm font-medium">
                    {doc.created_at
                      ? new Date(doc.created_at).toLocaleDateString()
                      : "—"}
                  </p>
                  <p className="text-xs text-muted-foreground">Uploaded</p>
                </div>
              </div>
              <div className="pt-2">
                <Badge variant="secondary">Indexed</Badge>
              </div>
            </CardContent>
          </Card>

          <div className="flex-1 min-h-0">
            <DocumentViewer documentId={id} filename={doc.filename} />
          </div>
        </div>
      ) : (
        <Card>
          <CardContent className="py-12 text-center">
            <p className="text-muted-foreground">Document not found</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
