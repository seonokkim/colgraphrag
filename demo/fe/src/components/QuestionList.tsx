import { MessageSquare } from 'lucide-react';
import type { QuestionSummary } from '@/types';

interface QuestionListProps {
  questions: QuestionSummary[];
  selectedQid: string | null;
  onSelect: (qid: string) => void;
}

const QCATE_COLORS: Record<string, string> = {
  YesNo: 'bg-blue-50 text-blue-600 border-blue-200',
  choose: 'bg-green-50 text-green-600 border-green-200',
  color: 'bg-amber-50 text-amber-600 border-amber-200',
  shape: 'bg-purple-50 text-purple-600 border-purple-200',
  text: 'bg-red-50 text-red-600 border-red-200',
  Others: 'bg-gray-50 text-gray-500 border-gray-200',
};

export function QuestionList({ questions, selectedQid, onSelect }: QuestionListProps) {
  return (
    <div className="flex flex-col gap-1 overflow-y-auto scrollbar-thin">
      {questions.map((q) => (
        <button
          key={q.qid}
          onClick={() => onSelect(q.qid)}
          className={`text-left px-3 py-2.5 rounded-lg transition-all text-[13px] ${
            selectedQid === q.qid
              ? 'bg-[#f0f0f0] text-[#1a1a1a]'
              : 'text-[#5a5a5a] hover:bg-[#f7f7f7]'
          }`}
          style={{ fontWeight: selectedQid === q.qid ? 500 : 400 }}
        >
          <div className="flex items-center gap-2 mb-1">
            <MessageSquare size={11} className="text-[#b0b0b0] shrink-0" />
            <span
              className={`text-[10px] px-1.5 py-0.5 rounded border ${
                QCATE_COLORS[q.qcate] || QCATE_COLORS.Others
              }`}
              style={{ fontWeight: 500 }}
            >
              {q.qcate}
            </span>
          </div>
          <p className="line-clamp-2 leading-snug">{q.question}</p>
        </button>
      ))}
    </div>
  );
}
