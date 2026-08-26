import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Jung Archive — Document Inspector",
  description:
    "Document intelligence workstation: parse, structure, chunk, retrieval and evidence inspection for the Jung corpus.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
