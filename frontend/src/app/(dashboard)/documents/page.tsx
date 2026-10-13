"use client";

import { useState, useCallback } from "react";
import { useDocuments, useUploadDocuments, useDeleteDocument } from "@/hooks/use-api";
import { useAuthStore } from "@/stores/auth-store";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { DocumentListSkeleton } from "@/components/shared/loading-states";
import {
  Upload,
  Search,
  Trash2,
  MoreHorizontal,
  FileText,
  Clock,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
  Filter,
  ArrowUpDown,
} from "lucide-react";
import { toast } from "sonner";

const SUPPORTED_TYPES = [
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "text/plain",
  "text/markdown",
  "text/html",
  "text/csv",
  "application/json",
];

const EXT_MAP: Record<string, string> = {
  pdf: "PDF",
  docx: "DOCX",
  txt: "TXT",
  md: "MD",
  html: "HTML",
  csv: "CSV",
  json: "JSON",
};

export default function DocumentsPage() {
  const { hasRole } = useAuthStore();
  const { data, isLoading, refetch } = useDocuments();
  const uploadMutation = useUploadDocuments();
  const deleteMutation = useDeleteDocument();
  const [searchQuery, setSearchQuery] = useState("");
  const [isDragOver, setIsDragOver] = useState(false);
  const [deleteDialog, setDeleteDialog] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<"filename" | "date">("date");

  if (!hasRole("admin")) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center space-y-2">
          <AlertCircle className="h-12 w-12 mx-auto text-muted-foreground" />
          <h2 className="text-lg font-semibold">Access Denied</h2>
          <p className="text-sm text-muted-foreground">Only administrators can manage documents.</p>
        </div>
      </div>
    );
  }

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const files = Array.from(e.dataTransfer.files);
      if (files.length > 0) {
        uploadMutation.mutate(files);
      }
    },
    [uploadMutation],
  );

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      uploadMutation.mutate(files);
    }
  };

  const handleDelete = (id: string) => {
    deleteMutation.mutate(id);
    setDeleteDialog(null);
  };

  const docs = data?.documents || [];
  const filtered = docs.filter(
    (d) => d.filename.toLowerCase().includes(searchQuery.toLowerCase()),
  );
  const sorted = [...filtered].sort((a, b) => {
    if (sortBy === "filename") return a.filename.localeCompare(b.filename);
    return (b.created_at || "").localeCompare(a.created_at || "");
  });

  return (
    <div className="flex flex-col h-full p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Documents</h1>
          <p className="text-sm text-muted-foreground">
            {data?.total_count || 0} document{(data?.total_count || 0) !== 1 ? "s" : ""} indexed
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4 mr-1" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Upload Zone */}
      <Card
        className={`border-2 border-dashed transition-colors ${
          isDragOver ? "border-primary bg-primary/5" : "border-muted-foreground/25"
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
      >
        <CardContent className="flex flex-col items-center justify-center py-12">
          <Upload className="h-10 w-10 text-muted-foreground mb-4" />
          <p className="text-sm font-medium mb-1">
            {uploadMutation.isPending
              ? "Uploading..."
              : "Drag & drop files here, or click to browse"}
          </p>
          <p className="text-xs text-muted-foreground mb-4">
            PDF, DOCX, TXT, Markdown, HTML, CSV, JSON — up to 50MB each
          </p>
          <input
            type="file"
            multiple
            accept=".pdf,.docx,.txt,.md,.html,.csv,.json"
            className="hidden"
            id="file-upload"
            onChange={handleFileSelect}
          />
          <Button variant="outline" size="sm">
            <label htmlFor="file-upload" className="cursor-pointer">
              Browse files
            </label>
          </Button>
        </CardContent>
      </Card>

      {/* Search & Filter */}
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search documents..."
            className="pl-9"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger render={<Button variant="outline" size="sm" />}>
            <ArrowUpDown className="h-4 w-4 mr-1" />
            Sort
          </DropdownMenuTrigger>
          <DropdownMenuContent>
            <DropdownMenuItem onClick={() => setSortBy("date")}>
              <Clock className="h-4 w-4 mr-2" />
              Date (newest)
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => setSortBy("filename")}>
              <FileText className="h-4 w-4 mr-2" />
              Filename
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Documents Table */}
      {isLoading ? (
        <DocumentListSkeleton />
      ) : sorted.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <FileText className="h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-sm text-muted-foreground">
              {searchQuery ? "No documents match your search" : "No documents uploaded yet"}
            </p>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Document</TableHead>
                <TableHead>Pages</TableHead>
                <TableHead>Chunks</TableHead>
                <TableHead>Uploaded</TableHead>
                <TableHead className="w-12" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {sorted.map((doc) => {
                const ext = doc.filename.split(".").pop()?.toLowerCase() || "";
                return (
                  <TableRow key={doc.document_id}>
                    <TableCell>
                      <div className="flex items-center gap-3">
                        <div className="flex h-9 w-9 items-center justify-center rounded bg-primary/10 text-xs font-bold text-primary">
                          {EXT_MAP[ext] || ext.toUpperCase().slice(0, 4)}
                        </div>
                        <div>
                          <p className="font-medium text-sm">{doc.filename}</p>
                          <p className="text-xs text-muted-foreground">
                            ID: {doc.document_id.slice(0, 8)}...
                          </p>
                        </div>
                      </div>
                    </TableCell>
                    <TableCell className="text-sm">{doc.total_pages}</TableCell>
                    <TableCell className="text-sm">{doc.total_chunks}</TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {doc.created_at
                        ? new Date(doc.created_at).toLocaleDateString()
                        : "—"}
                    </TableCell>
                    <TableCell>
                      <DropdownMenu>
                        <DropdownMenuTrigger
                          render={<Button variant="ghost" size="icon" className="h-8 w-8" />}
                        >
                          <MoreHorizontal className="h-4 w-4" />
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem
                            onClick={() =>
                              window.open(`/documents/${doc.document_id}`, "_blank")
                            }
                          >
                            <FileText className="h-4 w-4 mr-2" />
                            Preview
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            className="text-destructive"
                            onClick={() => setDeleteDialog(doc.document_id)}
                          >
                            <Trash2 className="h-4 w-4 mr-2" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </Card>
      )}

      {/* Delete Dialog */}
      <Dialog open={!!deleteDialog} onOpenChange={() => setDeleteDialog(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Document</DialogTitle>
            <DialogDescription>
              This will permanently remove the document and all its chunks. This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteDialog(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => deleteDialog && handleDelete(deleteDialog)}
              disabled={deleteMutation.isPending}
            >
              {deleteMutation.isPending ? "Deleting..." : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
