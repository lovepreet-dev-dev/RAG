"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { Plus, MessageSquare, Pin, Trash2, MoreHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Skeleton } from "@/components/ui/skeleton";
import {
  useChatSessions,
  useCreateChatSession,
  useDeleteChatSession,
  useUpdateChatSession,
} from "@/hooks/use-api";
import type { ChatSessionResponse } from "@/api/client";
import { cn } from "@/lib/utils";

interface ChatHistorySidebarProps {
  currentSessionId?: string;
  onNewChat?: () => void;
}

export function ChatHistorySidebar({
  currentSessionId,
  onNewChat,
}: ChatHistorySidebarProps) {
  const router = useRouter();
  const { data, isLoading } = useChatSessions();
  const createSession = useCreateChatSession();
  const deleteSession = useDeleteChatSession();
  const updateSession = useUpdateChatSession();

  const sessions = data?.sessions || [];
  const pinnedSessions = sessions.filter((s) => s.is_pinned);
  const otherSessions = sessions.filter((s) => !s.is_pinned);

  const handleNewChat = async () => {
    if (onNewChat) {
      onNewChat();
      return;
    }
    const session = await createSession.mutateAsync("New Chat");
    router.push(`/chat/${session.id}`);
  };

  const handleDelete = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm("Delete this chat?")) {
      await deleteSession.mutateAsync(sessionId);
      if (currentSessionId === sessionId) {
        router.push("/chat");
      }
    }
  };

  const handleTogglePin = async (sessionId: string, isPinned: boolean, e: React.MouseEvent) => {
    e.stopPropagation();
    await updateSession.mutateAsync({ sessionId, is_pinned: !isPinned });
  };

  const formatDate = (dateStr: string) => {
    if (!dateStr) return "";
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return "Today";
    if (diffDays === 1) return "Yesterday";
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  if (isLoading) {
    return (
      <div className="flex flex-col h-full border-r bg-muted/30">
        <div className="p-3">
          <Skeleton className="h-9 w-full" />
        </div>
        <Separator />
        <div className="flex-1 p-2 space-y-1">
          {Array.from({ length: 8 }).map((_, i) => (
            <Skeleton key={i} className="h-10 w-full" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full border-r bg-muted/30">
      <div className="p-3">
        <Button
          variant="outline"
          className="w-full justify-start gap-2"
          onClick={handleNewChat}
          disabled={createSession.isPending}
        >
          <Plus className="h-4 w-4" />
          New Chat
        </Button>
      </div>
      <Separator />
      <ScrollArea className="flex-1">
        <div className="p-2 space-y-1">
          {pinnedSessions.length > 0 && (
            <>
              <p className="px-2 py-1 text-xs font-medium text-muted-foreground uppercase">
                Pinned
              </p>
              {pinnedSessions.map((session) => (
                <SessionItem
                  key={session.id}
                  session={session}
                  isActive={session.id === currentSessionId}
                  onClick={() => router.push(`/chat/${session.id}`)}
                  onDelete={(e) => handleDelete(session.id, e)}
                  onTogglePin={(e) => handleTogglePin(session.id, session.is_pinned, e)}
                  formatDate={formatDate}
                />
              ))}
              {otherSessions.length > 0 && <Separator className="my-2" />}
            </>
          )}
          {otherSessions.length > 0 && (
            <>
              <p className="px-2 py-1 text-xs font-medium text-muted-foreground uppercase">
                Recent
              </p>
              {otherSessions.map((session) => (
                <SessionItem
                  key={session.id}
                  session={session}
                  isActive={session.id === currentSessionId}
                  onClick={() => router.push(`/chat/${session.id}`)}
                  onDelete={(e) => handleDelete(session.id, e)}
                  onTogglePin={(e) => handleTogglePin(session.id, session.is_pinned, e)}
                  formatDate={formatDate}
                />
              ))}
            </>
          )}
          {sessions.length === 0 && (
            <p className="px-2 py-8 text-center text-sm text-muted-foreground">
              No chats yet. Start a new one!
            </p>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}

function SessionItem({
  session,
  isActive,
  onClick,
  onDelete,
  onTogglePin,
  formatDate,
}: {
  session: ChatSessionResponse;
  isActive: boolean;
  onClick: () => void;
  onDelete: (e: React.MouseEvent) => void;
  onTogglePin: (e: React.MouseEvent) => void;
  formatDate: (date: string) => string;
}) {
  return (
    <div
      className={cn(
        "group flex items-center gap-2 rounded-md px-2 py-1.5 cursor-pointer hover:bg-accent/50 transition-colors",
        isActive && "bg-accent"
      )}
      onClick={onClick}
    >
      <MessageSquare className="h-4 w-4 shrink-0 text-muted-foreground" />
      <div className="flex-1 min-w-0">
        <p className="text-sm truncate">{session.title}</p>
        <p className="text-xs text-muted-foreground">
          {formatDate(session.updated_at)}
          {session.message_count > 0 && ` · ${session.message_count} msgs`}
        </p>
      </div>
      <DropdownMenu>
        <DropdownMenuTrigger
          className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity inline-flex items-center justify-center rounded-md hover:bg-accent"
          onClick={(e) => e.stopPropagation()}
        >
          <MoreHorizontal className="h-3 w-3" />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onClick={onTogglePin}>
            <Pin className="h-3 w-3 mr-2" />
            {session.is_pinned ? "Unpin" : "Pin"}
          </DropdownMenuItem>
          <DropdownMenuItem onClick={onDelete} className="text-destructive">
            <Trash2 className="h-3 w-3 mr-2" />
            Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
