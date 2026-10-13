import { NextRequest, NextResponse } from "next/server";

const publicRoutes = ["/login"];
const apiRoutes = ["/api"];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Allow API routes and public files
  if (
    apiRoutes.some((r) => pathname.startsWith(r)) ||
    pathname.startsWith("/_next") ||
    pathname.startsWith("/favicon") ||
    pathname.includes(".")
  ) {
    return NextResponse.next();
  }

  // Allow public routes
  if (publicRoutes.some((r) => pathname.startsWith(r))) {
    return NextResponse.next();
  }

  // Check for auth cookies (set by the auth store on login).
  // The access-token cookie lives 30 minutes, the refresh-token cookie 7 days.
  // Accept either: if only the refresh token survives, let the request through
  // so the app can mount and the API client's 401 interceptor can silently
  // refresh. Gating on auth_token alone hard-bounced users to /login after 30
  // minutes and made the refresh flow unreachable. The backend still validates
  // the JWT on every API call, so this gate is UX routing, not authorization.
  const token = request.cookies.get("auth_token")?.value;
  const refreshToken = request.cookies.get("refresh_token")?.value;

  if (!token && !refreshToken) {
    // No token cookie — redirect to login, preserving the intended URL
    const loginUrl = new URL("/login", request.url);
    loginUrl.searchParams.set("from", pathname);
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
