export const config = {
  api: {
    baseUrl: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  },
  app: {
    name: "Knowledge AI",
    description: "Enterprise AI Knowledge Platform",
    version: "1.0.0",
  },
  routes: {
    login: "/login",
    chat: "/chat",
    documents: "/documents",
    analytics: "/analytics",
    users: "/users",
    settings: "/settings",
    profile: "/profile",
  },
  roles: {
    admin: ["admin"] as const,
    employee: ["admin", "employee"] as const,
    viewer: ["admin", "employee", "viewer"] as const,
  },
} as const;
