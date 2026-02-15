/** @type {import('next').NextConfig} */
const withPWA = require('next-pwa')({
  dest: 'public',
  register: false, // We register manually for more control
  skipWaiting: false, // We handle this in our service worker
  disable: process.env.NODE_ENV === 'development', // Disable in dev
  buildExcludes: [/middleware-manifest\.json$/],
  publicExcludes: ['!sw.js', '!workbox-*.js'], // Keep our custom sw.js
});

const nextConfig = {
  reactStrictMode: true,
  // Standalone output for deployment
  output: 'standalone',
  swcMinify: true,
  // Transpile packages if needed
  transpilePackages: [],
  // Environment variables
  env: {
    CUSTOM_KEY: process.env.CUSTOM_KEY,
  },

  // PWA-specific headers
  async headers() {
    return [
      {
        source: '/manifest.json',
        headers: [
          {
            key: 'Content-Type',
            value: 'application/manifest+json',
          },
        ],
      },
      {
        source: '/sw.js',
        headers: [
          {
            key: 'Content-Type',
            value: 'application/javascript; charset=utf-8',
          },
          {
            key: 'Cache-Control',
            value: 'no-cache, no-store, must-revalidate',
          },
          {
            key: 'Service-Worker-Allowed',
            value: '/',
          },
        ],
      },
    ];
  },

  // Optimize images
  images: {
    formats: ['image/avif', 'image/webp'],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
  },

  // Webpack config for PWA assets
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        net: false,
        tls: false,
      };
    }
    return config;
  },
}

module.exports = withPWA(nextConfig)
