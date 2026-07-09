"use client";

import { cn } from "@/lib/utils";
import { ShieldCheck, Shield, AlertTriangle } from "lucide-react";

interface Props {
  level: "low" | "medium" | "high";
  score: number;
  className?: string;
}

const levelConfig = {
  high: {
    icon: ShieldCheck,
    color: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400",
    label: "High",
  },
  medium: {
    icon: Shield,
    color: "bg-amber-500/10 text-amber-600 dark:text-amber-400",
    label: "Medium",
  },
  low: {
    icon: AlertTriangle,
    color: "bg-red-500/10 text-red-600 dark:text-red-400",
    label: "Low",
  },
};

export function ConfidenceBadge({ level, score, className }: Props) {
  const config = levelConfig[level];
  const Icon = config.icon;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium",
        config.color,
        className,
      )}
    >
      <Icon className="h-3 w-3" />
      {config.label} ({(score * 100).toFixed(0)}%)
    </span>
  );
}
