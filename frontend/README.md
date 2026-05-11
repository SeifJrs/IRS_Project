# MarketMatch Next.js Frontend

This frontend talks to the Flask backend through a Next.js rewrite.

Run the backend from the project root:

```bash
python app.py
```

Run the frontend from this folder:

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

If Flask is not running on `http://127.0.0.1:5000`, set:

```bash
set FLASK_API_URL=http://127.0.0.1:5000
```
