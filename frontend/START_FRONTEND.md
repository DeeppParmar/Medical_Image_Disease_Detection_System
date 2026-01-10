# Start Frontend React App

## Quick Start

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Configure API URL (if backend is on different port)
Edit `frontend/src/config.ts`:
```typescript
export const API_BASE_URL = 'http://localhost:5000';  // Change if needed
```

### 3. Start Development Server
```bash
npm run dev
```

You should see:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:8080/
  ➜  Network: use --host to expose
```

### 4. Open Browser
Navigate to: http://localhost:8080

## Usage

1. Upload a chest X-ray image (PNG, JPG, JPEG)
2. Click "Analyze Image"
3. Wait for CheXNet model analysis
4. View results with disease detections and confidence scores

## Troubleshooting

### Backend Connection Failed
- Ensure backend is running on `http://localhost:5000`
- Check `frontend/src/config.ts` has correct API URL
- Check browser console for CORS errors

### Port Already in Use
- Change port in `vite.config.ts`:
  ```typescript
  server: {
    port: 8081,  // Change port
  }
  ```

### Module Not Found
- Delete `node_modules` and reinstall:
  ```bash
  rm -rf node_modules package-lock.json
  npm install
  ```

