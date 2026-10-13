"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { cn } from "@/lib/utils";
import { useAuthStore } from "@/stores/auth-store";
import { useChatSessions, useCreateChatSession } from "@/hooks/use-api";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Bot,
  MessageSquare,
  FileText,
  BarChart3,
  Users,
  Settings,
  Plus,
  Moon,
  Sun,
  LogOut,
  PanelLeftClose,
  PanelLeft,
} from "lucide-react";
import { useThemeStore } from "@/stores/theme-store";
import { useState } from "react";
import type { Role } from "@/types";

const navItems = [
  { href: "/chat", label: "Chat", icon: MessageSquare, roles: ["admin", "employee", "viewer"] as Role[] },
  { href: "/documents", label: "Documents", icon: FileText, roles: ["admin"] as Role[] },
  { href: "/analytics", label: "Analytics", icon: BarChart3, roles: ["admin"] as Role[] },
  { href: "/users", label: "Users", icon: Users, roles: ["admin"] as Role[] },
  { href: "/settings", label: "Settings", icon: Settings, roles: ["admin"] as Role[] },
];

export function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const { user, hasRole, logout } = useAuthStore();
  // Same API-backed source the chat page's sidebar uses, so the two lists
  // can't drift apart. Previously this read a local-only store, so sessions
  // shown here never matched the real ones on the server.
  const { data: sessionData } = useChatSessions();
  const createSession = useCreateChatSession();
  const { theme, setTheme } = useThemeStore();
  const [collapsed, setCollapsed] = useState(false);

  const sessions = sessionData?.sessions ?? [];
  const filteredNav = navItems.filter((item) => hasRole(...item.roles));

  const handleNewChat = async () => {
    const session = await createSession.mutateAsync("New Chat");
    router.push(`/chat/${session.id}`);
  };

  return (
    <aside
      className={cn(
        "flex flex-col border-r bg-sidebar text-sidebar-foreground transition-all duration-300",
        collapsed ? "w-16" : "w-72",
      )}
    >
      {/* Header */}
      <div className="flex h-14 items-center justify-between px-3">
        {!collapsed && (
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
              <Bot className="h-5 w-5 text-primary-foreground" />
            </div>
            <span className="text-sm font-semibold">Knowledge AI</span>
          </div>
        )}
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          onClick={() => setCollapsed(!collapsed)}
        >
          {collapsed ? (
            <PanelLeft className="h-4 w-4" />
          ) : (
            <PanelLeftClose className="h-4 w-4" />
          )}
        </Button>
      </div>

      <Separator />

      {/* New Chat Button */}
      <div className="p-2">
        <Button
          variant="outline"
          className={cn(
            "w-full justify-start gap-2",
            collapsed && "justify-center px-0",
          )}
          onClick={handleNewChat}
          disabled={createSession.isPending}
        >
          <Plus className="h-4 w-4" />
          {!collapsed && <span>New Chat</span>}
        </Button>
      </div>

      {/* Chat History */}
      {!collapsed && (
        <ScrollArea className="flex-1 px-2">
          <div className="space-y-1 py-2">
            {sessions.map((session) => (
              <div key={session.id} className="group relative">
                <button
                  onClick={() => router.push(`/chat/${session.id}`)}
                  className={cn(
                    "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors hover:bg-sidebar-accent",
                    pathname === `/chat/${session.id}` && "bg-sidebar-accent",
                  )}
                >
                  <MessageSquare className="h-4 w-4 shrink-0 text-muted-foreground" />
                  <span className="truncate">{session.title}</span>
                </button>
              </div>
            ))}
          </div>
        </ScrollArea>
      )}

      <Separator />

      {/* Navigation */}
      <nav className="space-y-1 p-2">
        {filteredNav.map((item) => {
          const Icon = item.icon;
          const isActive = pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors hover:bg-sidebar-accent",
                isActive && "bg-sidebar-accent font-medium",
                collapsed && "justify-center px-0",
              )}
            >
              <Icon className="h-4 w-4 shrink-0" />
              {!collapsed && <span>{item.label}</span>}
            </Link>
          );
        })}
      </nav>

      {/* Bottom section */}
      <div className="mt-auto space-y-1 p-2">
        <Separator className="mb-2" />

        {/* Theme Toggle */}
        <Button
          variant="ghost"
          className={cn(
            "w-full justify-start gap-2",
            collapsed && "justify-center px-0",
          )}
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        >
          {theme === "dark" ? (
            <Sun className="h-4 w-4" />
          ) : (
            <Moon className="h-4 w-4" />
          )}
          {!collapsed && (
            <span>{theme === "dark" ? "Light Mode" : "Dark Mode"}</span>
          )}
        </Button>

        {/* User */}
        <Link
          href="/profile"
          className={cn(
            "flex items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors hover:bg-sidebar-accent",
            collapsed && "justify-center px-0",
          )}
        >
          <div className="flex h-6 w-6 items-center justify-center rounded-full bg-muted text-xs font-medium">
            {user?.name?.charAt(0)?.toUpperCase() || "U"}
          </div>
          {!collapsed && (
            <div className="flex flex-col items-start">
              <span className="text-sm font-medium">{user?.name}</span>
              <span className="text-xs text-muted-foreground capitalize">
                {user?.role}
              </span>
            </div>
          )}
        </Link>

        {/* Logout */}
        <Button
          variant="ghost"
          className={cn(
            "w-full justify-start gap-2 text-muted-foreground",
            collapsed && "justify-center px-0",
          )}
          onClick={() => {
            logout();
            window.location.href = "/login";
          }}
        >
          <LogOut className="h-4 w-4" />
          {!collapsed && <span>Sign out</span>}
        </Button>
      </div>
    </aside>
  );
}
