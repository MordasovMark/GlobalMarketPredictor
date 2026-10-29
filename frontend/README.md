# Frontend

React 19 + Vite + Tailwind + Recharts. All data comes from the FastAPI backend (see the root README).

```
src/
  App.jsx                 shell: nav, search, routing, watchlist
  pages/                  HomeDashboard, StockDetailPage, SignalPortfolioPage
  components/             charts, rating panel, gauges, tables
  lib/api.js              the only place that calls the network
  lib/MarketDataContext   shared quotes / ratings / universe polling
  data/stocks.js          static names + descriptions (no prices)
```

`npm run dev` · `npm run lint` · `npm run build`. Set `VITE_API_BASE_URL` to point at another API.
