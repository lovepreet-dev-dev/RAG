"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Role, User } from "@/types";

function setCookie(name: string, value: string, maxAge: number) {
  if (typeof document !== "undefined") {
    const secure = !window.location.hostname.includes("localhost");
    const secureFlag = secure ? "; Secure" : "";
    document.cookie = `${name}=${encodeURIComponent(value)}; path=/; max-age=${maxAge}; SameSite=Lax${secureFlag}`;
  }
}

function removeCookie(name: string) {
  if (typeof document !== "undefined") {
    document.cookie = `${name}=; path=/; max-age=0`;
  }
}

interface AuthState {
  user: User | null;
  token: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  login: (user: User, accessToken: string, refreshToken: string) => void;
  logout: () => void;
  updateUser: (user: Partial<User>) => void;
  hasRole: (...roles: Role[]) => boolean;
  getRefreshToken: () => string | null;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      refreshToken: null,
      isAuthenticated: false,

      login: (user, accessToken, refreshToken) => {
        localStorage.setItem("access_token", accessToken);
        localStorage.setItem("refresh_token", refreshToken);
        localStorage.setItem("user", JSON.stringify(user));
        // Also set cookies for middleware auth checks
        setCookie("auth_token", accessToken, 30 * 60);
        setCookie("refresh_token", refreshToken, 7 * 24 * 60 * 60);
        set({ user, token: accessToken, refreshToken, isAuthenticated: true });
      },

      logout: () => {
        localStorage.removeItem("access_token");
        localStorage.removeItem("refresh_token");
        localStorage.removeItem("user");
        removeCookie("auth_token");
        removeCookie("refresh_token");
        set({ user: null, token: null, refreshToken: null, isAuthenticated: false });
      },

      updateUser: (partial) => {
        const current = get().user;
        if (current) {
          const updated = { ...current, ...partial };
          localStorage.setItem("user", JSON.stringify(updated));
          set({ user: updated });
        }
      },

      hasRole: (...roles) => {
        const user = get().user;
        return user ? roles.includes(user.role) : false;
      },

      getRefreshToken: () => {
        return get().refreshToken;
      },
    }),
    {
      name: "rag-auth",
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        refreshToken: state.refreshToken,
        isAuthenticated: state.isAuthenticated,
      }),
    },
  ),
);
