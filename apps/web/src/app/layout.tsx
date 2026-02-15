import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Jotty Management UI',
  description: 'Standalone Jotty Management Interface',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
