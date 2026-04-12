import { useState } from "react";
import { Header } from "./components/Header";
import { ChatSidebar } from "./components/ChatSidebar";
import { ChatInterface } from "./components/ChatInterface";
import { AdminPanel } from "./components/AdminPanel";
import { useSession } from "./hooks/useSession";

export default function App() {
  const sessionId = useSession();
  const [showAdmin, setShowAdmin] = useState(false);

  if (!sessionId) {
    return (
      <div className="flex items-center justify-center h-full">
        <span className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <Header onAdminClick={() => setShowAdmin((v) => !v)} showAdmin={showAdmin} />

      <div className="flex flex-1 overflow-hidden">
        <ChatSidebar />

        <main className="flex-1 flex overflow-hidden">
          <ChatInterface sessionId={sessionId} />
        </main>

        {showAdmin && <AdminPanel />}
      </div>
    </div>
  );
}
