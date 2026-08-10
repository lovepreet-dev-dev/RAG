"use client";

import { useState } from "react";
import { useSettings, useUpdateSettings } from "@/hooks/use-api";
import { useAuthStore } from "@/stores/auth-store";
import type { Settings } from "@/types";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Loader2, Save, AlertCircle } from "lucide-react";

export default function SettingsPage() {
  const { hasRole } = useAuthStore();
  const { data: settings, isLoading } = useSettings();
  const updateMutation = useUpdateSettings();
  const [form, setForm] = useState<Record<string, unknown>>({});

  if (!hasRole("admin")) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center space-y-2">
          <AlertCircle className="h-12 w-12 mx-auto text-muted-foreground" />
          <h2 className="text-lg font-semibold">Access Denied</h2>
          <p className="text-sm text-muted-foreground">Only administrators can access settings.</p>
        </div>
      </div>
    );
  }

  const get = (key: string) =>
    form[key] !== undefined ? form[key] : settings?.[key as keyof Settings];

  const set = (key: string, value: unknown) =>
    setForm((prev) => ({ ...prev, [key]: value }));

  // Each provider stores its model name under its own key; Ollama's model is
  // server-side only, so it has no editable field here.
  const provider = String(get("llm_provider") || "openai");
  const modelFields: Record<string, { key: string; placeholder: string }> = {
    openai: { key: "openai_model", placeholder: "gpt-4o-mini" },
    anthropic: { key: "anthropic_model", placeholder: "claude-3-5-sonnet-20241022" },
    openrouter: { key: "openrouter_model", placeholder: "meta-llama/llama-3.1-8b-instruct:free" },
  };
  const modelField = modelFields[provider];

  const handleSave = () => {
    updateMutation.mutate(form as Record<string, unknown>);
    setForm({});
  };

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin" />
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full p-6 space-y-6 max-w-4xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Settings</h1>
          <p className="text-sm text-muted-foreground">Configure your RAG system</p>
        </div>
        <Button onClick={handleSave} disabled={updateMutation.isPending}>
          {updateMutation.isPending ? (
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
          ) : (
            <Save className="h-4 w-4 mr-2" />
          )}
          Save Changes
        </Button>
      </div>

      {/* LLM Provider */}
      <Card>
        <CardHeader>
          <CardTitle>LLM Provider</CardTitle>
          <CardDescription>Configure the language model backend</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Provider</Label>
            <Select value={String(get("llm_provider") || "")} onValueChange={(v) => set("llm_provider", v)}>
              <SelectTrigger>
                <SelectValue placeholder="Select provider" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="openai">OpenAI</SelectItem>
                <SelectItem value="anthropic">Anthropic (Claude)</SelectItem>
                <SelectItem value="openrouter">OpenRouter</SelectItem>
                <SelectItem value="ollama">Ollama (Local)</SelectItem>
              </SelectContent>
            </Select>
          </div>
          {modelField ? (
            <div className="space-y-2">
              <Label>Model</Label>
              <Input
                value={String(get(modelField.key) || "")}
                onChange={(e) => set(modelField.key, e.target.value)}
                placeholder={modelField.placeholder}
              />
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">
              The Ollama model is configured on the server via the{" "}
              <code className="font-mono">OLLAMA_MODEL</code> environment variable.
            </p>
          )}
          <div className="space-y-2">
            <Label>Temperature</Label>
            <Input
              type="number"
              step="0.1"
              min="0"
              max="2"
              value={String(get("temperature") ?? 0.3)}
              onChange={(e) => set("temperature", parseFloat(e.target.value))}
            />
            <p className="text-xs text-muted-foreground">
              Lower values give more focused, deterministic answers.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Index */}
      <Card>
        <CardHeader>
          <CardTitle>Index</CardTitle>
          <CardDescription>Embedding model and vector store</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-2">
          <div className="space-y-1">
            <Label className="text-muted-foreground">Embedding Model</Label>
            <p className="text-sm font-medium">BAAI/bge-small-en-v1.5</p>
            <p className="text-xs text-muted-foreground">
              384 dimensions, runs locally — no embedding API cost.
            </p>
          </div>
          <div className="space-y-1">
            <Label className="text-muted-foreground">Vector Store</Label>
            <p className="text-sm font-medium">FAISS</p>
            <p className="text-xs text-muted-foreground">
              Persisted to disk. Changing these requires re-indexing all documents,
              so they are set on the server.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Retrieval Settings */}
      <Card>
        <CardHeader>
          <CardTitle>Retrieval</CardTitle>
          <CardDescription>Configure retrieval strategies and parameters</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Similarity Top K</Label>
              <Input
                type="number"
                value={String(get("similarity_top_k") || 10)}
                onChange={(e) => set("similarity_top_k", parseInt(e.target.value))}
              />
            </div>
            <div className="space-y-2">
              <Label>Hybrid Top K</Label>
              <Input
                type="number"
                value={String(get("hybrid_top_k") || 10)}
                onChange={(e) => set("hybrid_top_k", parseInt(e.target.value))}
              />
            </div>
            <div className="space-y-2">
              <Label>Chunk Size</Label>
              <Input
                type="number"
                value={String(get("chunk_size") || 512)}
                onChange={(e) => set("chunk_size", parseInt(e.target.value))}
              />
            </div>
            <div className="space-y-2">
              <Label>Chunk Overlap</Label>
              <Input
                type="number"
                value={String(get("chunk_overlap") || 128)}
                onChange={(e) => set("chunk_overlap", parseInt(e.target.value))}
              />
            </div>
            <div className="space-y-2">
              <Label>Score Threshold</Label>
              <Input
                type="number"
                step="0.05"
                min="0"
                max="1"
                value={String(get("score_threshold") ?? 0.3)}
                onChange={(e) => set("score_threshold", parseFloat(e.target.value))}
              />
            </div>
            <div className="space-y-2">
              <Label>Rerank Top K</Label>
              <Input
                type="number"
                value={String(get("rerank_top_k") || 5)}
                onChange={(e) => set("rerank_top_k", parseInt(e.target.value))}
              />
            </div>
            <div className="space-y-2">
              <Label>Max Context Tokens</Label>
              <Input
                type="number"
                value={String(get("max_context_tokens") || 3000)}
                onChange={(e) => set("max_context_tokens", parseInt(e.target.value))}
              />
            </div>
          </div>

          <Separator />

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Hybrid Search</Label>
                <p className="text-xs text-muted-foreground">
                  Combine dense vector search with BM25 keyword search
                </p>
              </div>
              <Switch
                checked={Boolean(get("enable_hybrid_search"))}
                onCheckedChange={(v) => set("enable_hybrid_search", v)}
              />
            </div>
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Reranker</Label>
                <p className="text-xs text-muted-foreground">Re-score results with a cross-encoder model</p>
              </div>
              <Switch
                checked={Boolean(get("enable_reranking"))}
                onCheckedChange={(v) => set("enable_reranking", v)}
              />
            </div>
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Answer Validation</Label>
                <p className="text-xs text-muted-foreground">
                  Check the generated answer against its sources for unsupported claims
                </p>
              </div>
              <Switch
                checked={Boolean(get("enable_answer_validation"))}
                onCheckedChange={(v) => set("enable_answer_validation", v)}
              />
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
