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
                <SelectItem value="gemini">Google Gemini</SelectItem>
                <SelectItem value="openrouter">OpenRouter</SelectItem>
                <SelectItem value="ollama">Ollama (Local)</SelectItem>
                <SelectItem value="lmstudio">LM Studio (Local)</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label>Model</Label>
            <Input
              value={String(get("openai_model") || "")}
              onChange={(e) => set("openai_model", e.target.value)}
              placeholder="gpt-4o-mini"
            />
          </div>
        </CardContent>
      </Card>

      {/* Embeddings */}
      <Card>
        <CardHeader>
          <CardTitle>Embeddings</CardTitle>
          <CardDescription>Configure the embedding model</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <Label>Embedding Model</Label>
            <Select value={String(get("embedding_model") || "")} onValueChange={(v) => set("embedding_model", v)}>
              <SelectTrigger>
                <SelectValue placeholder="Select model" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="bge-small">BGE Small</SelectItem>
                <SelectItem value="bge-large">BGE Large</SelectItem>
                <SelectItem value="minilm">MiniLM</SelectItem>
                <SelectItem value="openai">OpenAI</SelectItem>
                <SelectItem value="nomic">Nomic</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Vector Store */}
      <Card>
        <CardHeader>
          <CardTitle>Vector Store</CardTitle>
          <CardDescription>Choose your vector database backend</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <Label>Vector Store</Label>
            <Select value={String(get("vector_store") || "")} onValueChange={(v) => set("vector_store", v)}>
              <SelectTrigger>
                <SelectValue placeholder="Select store" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="faiss">FAISS</SelectItem>
                <SelectItem value="chroma">Chroma</SelectItem>
                <SelectItem value="qdrant">Qdrant</SelectItem>
                <SelectItem value="pinecone">Pinecone</SelectItem>
              </SelectContent>
            </Select>
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
              <Label>MMR Top K</Label>
              <Input
                type="number"
                value={String(get("mmr_top_k") || 10)}
                onChange={(e) => set("mmr_top_k", parseInt(e.target.value))}
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
          </div>

          <Separator />

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>MMR (Maximal Marginal Relevance)</Label>
                <p className="text-xs text-muted-foreground">Enable diversity-aware retrieval</p>
              </div>
              <Switch
                checked={Boolean(get("enable_mmr"))}
                onCheckedChange={(v) => set("enable_mmr", v)}
              />
            </div>
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Hybrid Search</Label>
                <p className="text-xs text-muted-foreground">Combine semantic and keyword search</p>
              </div>
              <Switch
                checked={Boolean(get("enable_hybrid_search"))}
                onCheckedChange={(v) => set("enable_hybrid_search", v)}
              />
            </div>
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Multi-Query</Label>
                <p className="text-xs text-muted-foreground">Generate alternative queries for low-confidence results</p>
              </div>
              <Switch
                checked={Boolean(get("enable_multi_query"))}
                onCheckedChange={(v) => set("enable_multi_query", v)}
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
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
