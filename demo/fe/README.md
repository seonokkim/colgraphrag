# ColGraphRAG WebQA Demo Frontend

React + Vite + TypeScript chat-style UI for WebQA results.

## Quick Start

```bash
cd demo/fe
npm install
npm run dev
```

The dev script binds **0.0.0.0:5173** so the UI is reachable from RunPod HTTP proxy or LAN.

**Note:** Start the backend first (`demo/be/`, default port **8000**). The FE calls `/api` and `/health`; Vite proxies them to `http://127.0.0.1:8000`.

### RunPod / Remote SSH

1. Use **Node 18+** (e.g. `nvm install --lts`). Ubuntu `apt` Node 12 is too old for Vite 6.
2. Expose HTTP port **5173** in the pod; open  
   `https://<POD_ID>-5173.proxy.runpod.net` (see [Runpod networking](https://docs.runpod.io/pods/networking)).
3. Optional: set backend URL for the proxy in `demo/fe/.env.local`:

   ```bash
   VITE_PROXY_BE_TARGET=http://127.0.0.1:8000
   ```

4. To expose the API docs directly, run the BE with `--host 0.0.0.0` and expose port **8000**:

   ```bash
   cd demo/be
   python server.py --host 0.0.0.0 --port 8000
   ```

**Important:** Do not strip the `/api` prefix in the Vite proxy. FastAPI routes are `/api/run/...`, `/api/questions/...`, etc.

### BE bind address

```bash
cd demo/be
python server.py --host 0.0.0.0 --port 8000
# or
uvicorn server:app --host 0.0.0.0 --port 8000
```

## Features

- Chat-style Q&A display
- Score cards (QA-FL / QA-Acc / QA)
- Question list with category filters
- Gold vs Predicted answer comparison
- Retrieval ranking display
- Knowledge graph visualization (react-force-graph-2d)
- Evidence image viewer

## Tech Stack

- React 19
- Vite 6
- TypeScript
- Tailwind CSS 4
- React Router 7
- react-force-graph-2d

## Directory Structure

```
demo/fe/
  index.html
  package.json
  vite.config.ts         # Proxy /api -> :8000
  tsconfig.json
  src/
    main.tsx             # Entry point
    App.tsx              # Router setup
    api/
      client.ts          # API fetch wrapper
    components/
      ScoreCards.tsx     # QA metrics cards
      QuestionList.tsx   # Sidebar question list
      ChatBubble.tsx     # Chat message bubble
      AnswerComparison.tsx
      RetrievalList.tsx
      ImageViewer.tsx
      GraphViewer.tsx
    pages/
      ChatPage.tsx       # Main chat interface
    styles/
      index.css          # Tailwind + custom styles
    types/
      index.ts           # TypeScript types
```

## Build

```bash
npm run build
```

Output in `dist/` folder.
