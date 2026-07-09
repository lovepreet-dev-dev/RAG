"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/stores/auth-store";
import { Button } from "@/components/ui/button";
import { LogIn, Bot } from "lucide-react";

export default function UnauthorizedPage() {
  const router = useRouter();
  const { isAuthenticated } = useAuthStore();

  useEffect(() => {
    if (isAuthenticated) {
      router.replace("/chat");
    }
  }, [isAuthenticated, router]);

  return (
    <div className="flex h-screen flex-col items-center justify-center gap-6">
      <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary">
        <Bot className="h-8 w-8 text-primary-foreground" />
      </div>
      <div className="text-center space-y-2">
        <h1 className="text-2xl font-bold">Welcome to Knowledge AI</h1>
        <p className="text-muted-foreground">
          Sign in to start chatting with your knowledge base
        </p>
      </div>
      <Button onClick={() => router.push("/login")} size="lg">
        <LogIn className="mr-2 h-4 w-4" />
        Sign In
      </Button>
    </div>
  );
}
