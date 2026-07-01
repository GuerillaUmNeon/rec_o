import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "rec_o",
  description:
    "Search manually or use a ListenBrainz profile to generate artist and album recommendations.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
