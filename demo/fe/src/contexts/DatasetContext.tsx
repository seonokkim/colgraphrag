import { createContext, useContext, useState } from 'react';
import type { ReactNode } from 'react';
import type { DatasetKey } from '@/types';

interface DatasetContextValue {
  dataset: DatasetKey;
  setDataset: (d: DatasetKey) => void;
}

export const DatasetContext = createContext<DatasetContextValue>({
  dataset: 'webqa',
  setDataset: () => {},
});

export function DatasetProvider({ children }: { children: ReactNode }) {
  const [dataset, setDataset] = useState<DatasetKey>('webqa');
  return (
    <DatasetContext.Provider value={{ dataset, setDataset }}>
      {children}
    </DatasetContext.Provider>
  );
}

export function useDataset() {
  return useContext(DatasetContext);
}
