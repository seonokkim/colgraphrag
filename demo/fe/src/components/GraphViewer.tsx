import { useEffect, useRef, useCallback, useState } from 'react';
import { GitBranch } from 'lucide-react';
import type { GraphData } from '@/types';

interface GraphViewerProps {
  data: GraphData | null;
  loading?: boolean;
}

interface NodeObject {
  id: string;
  label: string;
  type: string | null;
  x?: number;
  y?: number;
}


const NODE_COLORS: Record<string, string> = {
  building: '#3b82f6',
  person: '#22c55e',
  movie: '#0ea5e9',
  image: '#f59e0b',
  text: '#8b5cf6',
  default: '#6b7280',
};

export function GraphViewer({ data, loading }: GraphViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 400, height: 224 });
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [ForceGraph, setForceGraph] = useState<React.ComponentType<any> | null>(null);

  useEffect(() => {
    import('react-force-graph-2d').then((mod) => {
      setForceGraph(() => mod.default);
    });
  }, []);

  useEffect(() => {
    if (loading || !data || data.nodes.length === 0) return;
    const el = containerRef.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    function measure() {
      const w = el.clientWidth;
      if (w > 0) setDimensions((d) => (d.width === Math.floor(w) ? d : { width: Math.floor(w), height: 224 }));
    }
    measure();
    const ro = new ResizeObserver(() => measure());
    ro.observe(el);
    return () => ro.disconnect();
  }, [loading, data]);

  const getNodeColor = useCallback((node: NodeObject) => {
    const type = node.type?.toLowerCase() || '';
    for (const [key, color] of Object.entries(NODE_COLORS)) {
      if (key !== 'default' && type.includes(key)) return color;
    }
    return NODE_COLORS.default;
  }, []);

  if (loading) {
    return (
      <div className="border-t border-[#f0f0f0] pt-3">
        <div className="h-56 min-h-[224px] w-full rounded-xl bg-[#fafafa] border border-[#f0f0f0] flex items-center justify-center">
          <div className="flex gap-1.5 items-center">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="w-1.5 h-1.5 bg-[#b0b0b0] rounded-full animate-bounce"
                style={{ animationDelay: `${i * 0.15}s` }}
              />
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (!data || data.nodes.length === 0) return null;

  const graphData = {
    nodes: data.nodes.map((n) => ({
      id: n.id,
      label: n.entity_name || n.id,
      type: n.node_type,
    })),
    links: data.edges.map((e) => ({
      source: e.source,
      target: e.target,
      label: e.description,
    })),
  };

  return (
    <div className="border-t border-[#f0f0f0] pt-3">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5">
          <GitBranch size={12} className="text-[#9a9a9a]" />
          <span className="text-[11px] text-[#9a9a9a]" style={{ fontWeight: 600 }}>
            Knowledge Graph
          </span>
        </div>
        <span className="text-[10px] text-[#b0b0b0]">
          {data.node_count} nodes, {data.edge_count} edges
        </span>
      </div>
      <div
        ref={containerRef}
        className="h-56 min-h-[224px] w-full rounded-xl bg-[#fafafa] border border-[#f0f0f0] overflow-hidden"
      >
        {ForceGraph ? (
          <ForceGraph
            graphData={graphData}
            nodeLabel="label"
            nodeColor={getNodeColor}
            linkDirectionalArrowLength={3}
            linkColor={() => '#d4d4d4'}
            linkLabel={(link: { label: string | null }) => link.label}
            width={Math.max(dimensions.width, 320)}
            height={224}
            backgroundColor="#fafafa"
          />
        ) : (
          <div className="h-full flex items-center justify-center text-[12px] text-[#b0b0b0]">
            Loading graph...
          </div>
        )}
      </div>
    </div>
  );
}
