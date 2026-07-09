"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useChatStore } from "@/stores/chat-store";
import { useDocuments } from "@/hooks/use-api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { MessageSquare, FileText, Settings, BarChart3, Users, Search } from "lucide-react";

export function SearchDialog() {
  const [open, setOpen] = useState(false);
  const router = useRouter();
  const { sessions } = useChatStore();
  const { data: docs } = useDocuments();

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((prev) => !prev);
      }
    };
    document.addEventListener("keydown", down);
    return () => document.removeEventListener("keydown", down);
  }, []);

  const navigate = useCallback(
    (href: string) => {
      setOpen(false);
      router.push(href);
    },
    [router],
  );

  return (
    <>
      <Button
        variant="outline"
        className="flex items-center gap-2 text-muted-foreground"
        onClick={() => setOpen(true)}
      >
        <Search className="h-4 w-4" />
        <span className="hidden md:inline">Search documents...</span>
        <kbd className="pointer-events-none hidden select-none rounded-md border bg-muted px-1.5 text-[10px] font-medium md:inline">
          ⌘K
        </kbd>
      </Button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="p-0 max-w-lg">
          <DialogTitle className="sr-only">Search</DialogTitle>
          <Command>
            <CommandInput placeholder="Search documents, chats, pages..." />
            <CommandList>
              <CommandEmpty>No results found.</CommandEmpty>

              <CommandGroup heading="Pages">
                <CommandItem onSelect={() => navigate("/chat")}>
                  <MessageSquare className="mr-2 h-4 w-4" />
                  Chat
                </CommandItem>
                <CommandItem onSelect={() => navigate("/documents")}>
                  <FileText className="mr-2 h-4 w-4" />
                  Documents
                </CommandItem>
                <CommandItem onSelect={() => navigate("/analytics")}>
                  <BarChart3 className="mr-2 h-4 w-4" />
                  Analytics
                </CommandItem>
                <CommandItem onSelect={() => navigate("/users")}>
                  <Users className="mr-2 h-4 w-4" />
                  Users
                </CommandItem>
                <CommandItem onSelect={() => navigate("/settings")}>
                  <Settings className="mr-2 h-4 w-4" />
                  Settings
                </CommandItem>
              </CommandGroup>

              {sessions.length > 0 && (
                <CommandGroup heading="Recent Chats">
                  {sessions.slice(0, 5).map((session) => (
                    <CommandItem
                      key={session.id}
                      onSelect={() => navigate(`/chat/${session.id}`)}
                    >
                      <MessageSquare className="mr-2 h-4 w-4" />
                      {session.title}
                    </CommandItem>
                  ))}
                </CommandGroup>
              )}

              {docs?.documents && docs.documents.length > 0 && (
                <CommandGroup heading="Documents">
                  {docs.documents.slice(0, 5).map((doc) => (
                    <CommandItem
                      key={doc.document_id}
                      onSelect={() => navigate(`/documents/${doc.document_id}`)}
                    >
                      <FileText className="mr-2 h-4 w-4" />
                      {doc.filename}
                    </CommandItem>
                  ))}
                </CommandGroup>
              )}
            </CommandList>
          </Command>
        </DialogContent>
      </Dialog>
    </>
  );
}
