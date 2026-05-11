const flaskApiUrl = process.env.FLASK_API_URL || "http://127.0.0.1:5000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/flask/:path*",
        destination: `${flaskApiUrl}/api/:path*`
      }
    ];
  }
};

export default nextConfig;
