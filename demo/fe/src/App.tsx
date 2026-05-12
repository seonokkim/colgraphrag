import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { DatasetProvider } from '@/contexts/DatasetContext';
import { ChatPage } from '@/pages/ChatPage';

export default function App() {
  return (
    <DatasetProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<ChatPage />} />
        </Routes>
      </BrowserRouter>
    </DatasetProvider>
  );
}
