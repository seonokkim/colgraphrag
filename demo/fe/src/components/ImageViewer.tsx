import { Image as ImageIcon } from 'lucide-react';
import type { GoldFact, DatasetKey } from '@/types';
import { api } from '@/api/client';

interface ImageViewerProps {
  facts: GoldFact[];
  dataset?: DatasetKey;
}

export function ImageViewer({ facts, dataset = 'webqa' }: ImageViewerProps) {
  const imageFacts = facts.filter((f) => f.fact_type === 'image');
  if (imageFacts.length === 0) return null;

  return (
    <div className="border-t border-[#f0f0f0] pt-3">
      <div className="flex items-center gap-1.5 mb-2">
        <ImageIcon size={12} className="text-[#9a9a9a]" />
        <span className="text-[11px] text-[#9a9a9a]" style={{ fontWeight: 600 }}>
          Evidence Images
        </span>
      </div>
      <div className="grid grid-cols-2 gap-2">
        {imageFacts.map((img) => (
          <div key={img.id} className="relative group rounded-lg overflow-hidden border border-[#f0f0f0]">
            <img
              src={api.getImageUrl(img.id, dataset)}
              alt={img.title || `Image ${img.id}`}
              className="w-full h-28 object-cover bg-[#fafafa]"
              onError={(e) => {
                const target = e.target as HTMLImageElement;
                target.style.display = 'none';
              }}
            />
            <div className="absolute bottom-0 left-0 right-0 bg-white/80 backdrop-blur-sm text-[10px] text-[#5a5a5a] px-2 py-1 opacity-0 group-hover:opacity-100 transition-opacity">
              {img.id}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
