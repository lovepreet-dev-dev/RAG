"use client";

import { useState } from "react";
import { ChatView } from "@/components/chat/chat-view";
import { ChatHistorySidebar } from "@/components/chat/chat-history-sidebar";

export default function ChatPage() {
  const [currentSessionId, setCurrentSessionId] = useState<string | undefined>();

  return (
    <div className="flex h-full">
      <ChatHistorySidebar
        currentSessionId={currentSessionId}
        onNewChat={() => setCurrentSessionId(undefined)}
      />
      <div className="flex-1 min-w-0">
        <ChatView sessionId={currentSessionId} />
      </div>
    </div>
  );
}
