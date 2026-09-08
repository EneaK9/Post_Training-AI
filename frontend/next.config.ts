import type { NextConfig } from "next";

// The API is same-origin from the browser's point of view: /api/* is rewritten to the
// FastAPI server so the session cookie needs no CORS dance.
const API_URL = process.env.API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/api/:path*` }];
  },
};

export default nextConfig;
