"use client";

import { useState, useEffect } from "react";
import { AlertCircle, Wifi, WifiOff } from "lucide-react";
import { cn } from "@/lib/utils";

export function OfflineBanner() {
  const [isOnline, setIsOnline] = useState(true);
  const [show, setShow] = useState(false);

  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      setShow(false);
    };
    const handleOffline = () => {
      setIsOnline(false);
      setShow(true);
    };

    setIsOnline(navigator.onLine);
    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  if (!show) return null;

  return (
    <div
      className={cn(
        "fixed bottom-4 left-1/2 -translate-x-1/2 z-50",
        "flex items-center gap-2 rounded-lg border bg-background px-4 py-2 shadow-lg",
        "animate-in fade-in slide-in-from-bottom-2"
      )}
    >
      <WifiOff className="h-4 w-4 text-muted-foreground" />
      <span className="text-sm font-medium">You&apos;re offline</span>
      <span className="text-xs text-muted-foreground">
        — Some features may be unavailable
      </span>
    </div>
  );
}
