"use client";

import { useStats, useHealth, useDocuments } from "@/hooks/use-api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatsCardSkeleton } from "@/components/shared/loading-states";
import {
  FileText,
  Layers,
  MessageSquare,
  Users,
  ShieldCheck,
  Clock,
  Cpu,
  Zap,
  Activity,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";

const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884d8"];

export default function AnalyticsPage() {
  const { data: stats, isLoading: statsLoading } = useStats();
  const { data: health, isLoading: healthLoading } = useHealth();
  const { data: docData } = useDocuments();

  const cards = [
    {
      title: "Total Documents",
      value: stats?.total_documents ?? "—",
      icon: FileText,
      description: "Indexed documents",
    },
    {
      title: "Indexed Chunks",
      value: stats?.total_chunks ?? "—",
      icon: Layers,
      description: "Total text chunks",
    },
    {
      title: "Queries Today",
      value: stats?.total_queries ?? 0,
      icon: MessageSquare,
      description: "Total queries made",
    },
    {
      title: "Avg Latency",
      value: stats?.avg_latency_ms ? `${stats.avg_latency_ms.toFixed(0)}ms` : "—",
      icon: Clock,
      description: "Average response time",
    },
    {
      title: "Total Cost",
      value: stats?.total_cost ? `$${stats.total_cost.toFixed(4)}` : "$0.00",
      icon: Zap,
      description: "API cost estimate",
    },
    {
      title: "Tokens Used",
      value: stats?.total_tokens_used
        ? stats.total_tokens_used > 1000
          ? `${(stats.total_tokens_used / 1000).toFixed(1)}k`
          : stats.total_tokens_used
        : "—",
      icon: Activity,
      description: "Total tokens consumed",
    },
    {
      title: "Embedding Model",
      value: stats?.embedding_model || "—",
      icon: Cpu,
      description: "Current embedding model",
    },
    {
      title: "LLM Provider",
      value: stats?.llm_provider
        ? stats.llm_provider.charAt(0).toUpperCase() + stats.llm_provider.slice(1)
        : "—",
      icon: Zap,
      description: "Active LLM backend",
    },
  ];

  // Prepare chart data from documents
  const docChartData = (docData?.documents || []).slice(0, 10).map((doc) => ({
    name: doc.filename.length > 15 ? doc.filename.slice(0, 15) + "..." : doc.filename,
    chunks: doc.total_chunks,
    pages: doc.total_pages,
  }));

  // Prepare pie chart data for document distribution
  const pieData = (docData?.documents || []).slice(0, 5).map((doc) => ({
    name: doc.filename,
    value: doc.total_chunks,
  }));

  return (
    <div className="flex flex-col h-full p-6 space-y-6 overflow-auto">
      <div>
        <h1 className="text-2xl font-bold">Analytics</h1>
        <p className="text-sm text-muted-foreground">
          System performance and usage metrics
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {statsLoading
          ? Array.from({ length: 8 }).map((_, i) => <StatsCardSkeleton key={i} />)
          : cards.map((card) => {
              const Icon = card.icon;
              return (
                <Card key={card.title}>
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div className="space-y-1">
                        <p className="text-sm text-muted-foreground">{card.title}</p>
                        <p className="text-2xl font-bold">{card.value}</p>
                        <p className="text-xs text-muted-foreground">{card.description}</p>
                      </div>
                      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                        <Icon className="h-5 w-5 text-primary" />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
      </div>

      {/* Charts */}
      {docChartData.length > 0 && (
        <div className="grid gap-6 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Document Chunks Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={docChartData}>
                  <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="chunks" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Document Size Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) =>
                      `${(name ?? "").slice(0, 10)}... ${((percent ?? 0) * 100).toFixed(0)}%`
                    }
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {pieData.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      )}

      {/* System Info */}
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">System Status</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Status</span>
              <span className="flex items-center gap-2 text-sm font-medium">
                <span className="h-2 w-2 rounded-full bg-emerald-500" />
                {health?.status || "Unknown"}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Version</span>
              <span className="text-sm font-medium">{health?.version || "—"}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Vector Store</span>
              <span className="text-sm font-medium">{health?.vector_store || stats?.vector_store_type || "—"}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Documents Indexed</span>
              <span className="text-sm font-medium">{health?.documents_indexed ?? "—"}</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Top Documents</CardTitle>
          </CardHeader>
          <CardContent>
            {docData?.documents && docData.documents.length > 0 ? (
              <div className="space-y-3">
                {docData.documents.slice(0, 5).map((doc) => (
                  <div key={doc.document_id} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <FileText className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm truncate max-w-[200px]">{doc.filename}</span>
                    </div>
                    <span className="text-xs text-muted-foreground">{doc.total_chunks} chunks</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground text-center py-4">No documents yet</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
