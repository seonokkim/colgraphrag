import { CheckCircle2, XCircle } from 'lucide-react';

interface AnswerComparisonProps {
  goldAnswers: string[];
  predictedAnswer: string | null;
  keywordsAnswer: string | null;
}

export function AnswerComparison({
  goldAnswers,
  predictedAnswer,
  keywordsAnswer,
}: AnswerComparisonProps) {
  return (
    <div className="space-y-3">
      <div>
        <div className="flex items-center gap-1.5 mb-1.5">
          <CheckCircle2 size={12} className="text-[#22c55e]" />
          <span className="text-[11px] text-[#9a9a9a]" style={{ fontWeight: 600 }}>
            Predicted
          </span>
        </div>
        <div className="text-[14px] leading-relaxed text-[#1a1a1a]">
          {predictedAnswer || (
            <span className="text-[#b0b0b0] italic">No prediction</span>
          )}
        </div>
      </div>

      <div className="border-t border-[#f0f0f0] pt-3">
        <div className="flex items-center gap-1.5 mb-1.5">
          <XCircle size={12} className="text-[#9a9a9a]" />
          <span className="text-[11px] text-[#9a9a9a]" style={{ fontWeight: 600 }}>
            Gold Answer
          </span>
        </div>
        <div className="rounded-lg bg-[#f0fdf4] border border-[#bbf7d0] px-3 py-2 text-[13px] text-[#15803d] leading-relaxed">
          {goldAnswers.map((a, i) => (
            <p key={i}>{a}</p>
          ))}
        </div>
        {keywordsAnswer && (
          <p className="mt-1.5 text-[11px] text-[#b0b0b0]">
            Keywords: <span className="text-[#7a7a7a]">{keywordsAnswer}</span>
          </p>
        )}
      </div>
    </div>
  );
}
