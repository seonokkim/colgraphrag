import { Bot, User } from 'lucide-react';
import { ReactNode } from 'react';

interface ChatBubbleProps {
  role: 'user' | 'assistant';
  children: ReactNode;
}

export function ChatBubble({ role, children }: ChatBubbleProps) {
  const isUser = role === 'user';

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div
        className={`w-7 h-7 rounded-full flex items-center justify-center shrink-0 mt-0.5 ${
          isUser ? 'bg-[#f0f0f0]' : 'bg-[#1a1a1a]'
        }`}
      >
        {isUser ? (
          <User size={13} color="#4a4a4a" />
        ) : (
          <Bot size={13} color="white" />
        )}
      </div>

      <div className={`max-w-[82%] flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
        <div
          className={`rounded-2xl px-4 py-3 text-[14px] leading-relaxed ${
            isUser
              ? 'bg-[#1a1a1a] text-white rounded-tr-sm'
              : 'bg-[#f9f9f9] text-[#1a1a1a] border border-[#eeeeee] rounded-tl-sm'
          }`}
        >
          {children}
        </div>
      </div>
    </div>
  );
}
