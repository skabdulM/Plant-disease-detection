import {
  ClerkProvider,
  SignInButton,
  SignedIn,
  SignedOut,
  UserButton
} from '@clerk/nextjs'
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Sidebar from "@/components/Sidebar";
import Navbar from "@/components/Navbar";
import { FloatingDockDemo } from "./FloatingDock";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Create Next App",
  description: "Generated by create next app",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <ClerkProvider>

      <html lang="en">
        <body className={inter.className}>
          <div className="min-h-screen flex flex-col  ">
            <Navbar />
            <div className="flex">
              <Sidebar />
              <main className="flex-1 p-4">
                {children}
              </main>
            </div>
            <footer>
              <div className="dock fixed bottom-5 left-1/2 transform -translate-x-1/2 ">
                {/* <FloatingDockDemo /> */}
              </div>

            </footer>
            {/* <iframe src="https://bot.elephant.ai/46b68a4d-5021-4644-80c7-74678d6c5d06" width="100%" style={{ height: "100%", minHeight: "700px" }} frameBorder="0"></iframe> */}
          </div>
        </body>
      </html>
    </ClerkProvider>
  );
}
