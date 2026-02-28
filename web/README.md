# ARKE Physical AI - Web Frontend

React-based web frontend for ARKE Physical AI.

## Getting Started

### Install Dependencies

```bash
npm install
```

### Development

Run the development server:

```bash
npm run dev
```

The application will be available at `http://localhost:3000`

### Build

Build for production:

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

## Tech Stack

- **React** - UI framework
- **Vite** - Build tool and dev server
- **Axios** - HTTP client for API requests
- **ESLint** - Code linting

## API Integration

The frontend is configured to proxy API requests to `http://localhost:8000`. Update the proxy configuration in `vite.config.js` if your API runs on a different port.
