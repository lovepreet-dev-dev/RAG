"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Role, User } from "@/types";

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
        set({ user, token: accessToken, refreshToken, isAuthenticated: true });
      },

      logout: () => {
        localStorage.removeItem("access_token");
        localStorage.removeItem("refresh_token");
        localStorage.removeItem("user");
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
