import { BookOpen } from 'lucide-react';
import type { RetrievalItem } from '@/types';

interface RetrievalListProps {
  items: RetrievalItem[];
}

export function RetrievalList({ items }: RetrievalListProps) {
  if (items.length === 0) return null;

  return (
    <div className="border-t border-[#f0f0f0] pt-3">
      <div className="flex items-center gap-1.5 mb-2">
        <BookOpen size={12} className="text-[#9a9a9a]" />
        <span className="text-[11px] text-[#9a9a9a]" style={{ fontWeight: 600 }}>
          Sources ({items.length})
        </span>
      </div>
      <div className="space-y-1">
        {items.map((item, idx) => (
          <div
            key={item.id}
            className="flex items-center justify-between rounded-lg bg-[#fafafa] border border-[#f0f0f0] px-3 py-2 text-[12px]"
          >
            <div className="flex items-center gap-2 min-w-0">
              <span className="text-[#b0b0b0] shrink-0 w-5 text-right">
                #{item.rank ?? idx + 1}
              </span>
              <span className="text-[#4a4a4a] font-mono truncate">
                {item.id}
              </span>
            </div>
            <span
              className={`font-mono shrink-0 ml-2 ${
                item.score > 0.5 ? 'text-[#22c55e]' : 'text-[#b0b0b0]'
              }`}
            >
              {item.score.toFixed(3)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
