import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "A call with someone you have lost",
  description: "A live video call with a synthetic recreation of someone who has died.",
};

export const viewport: Viewport = {
  // The interface is light by default and dark by choice; the browser chrome
  // should follow it rather than sit against it.
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#000000" },
  ],
};

// Runs before the first paint. Without it, somebody who has chosen dark sees a
// white page for a frame on every navigation, which on this product reads as
// something going wrong. Kept to one line for the same reason.
//
// The attribute it writes is not in the server's markup, which is what
// suppressHydrationWarning below is for: React must be told that this one
// element is expected to differ, or it discards the correction along with
// every other client fix inside the same boundary.
const APPEARANCE = `try{var a=localStorage.getItem("appearance");if(a==="light"||a==="dark")document.documentElement.dataset.theme=a}catch(e){}`;

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: APPEARANCE }} />
      </head>
      <body className="min-h-full bg-surface text-label">{children}</body>
    </html>
  );
}
