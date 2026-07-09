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

  // Check for token in localStorage (client-side auth)
  // The middleware runs on the server, so we can't access localStorage.
  // Auth checking is done client-side via the auth store and layout guards.
  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
