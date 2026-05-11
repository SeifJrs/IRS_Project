"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BadgeDollarSign,
  BarChart3,
  BookOpen,
  Boxes,
  Brain,
  Building2,
  Car,
  ChartNoAxesCombined,
  ChevronRight,
  CircleDollarSign,
  Dumbbell,
  Fingerprint,
  Gauge,
  HeartPulse,
  Home,
  Layers3,
  Loader2,
  Monitor,
  Network,
  PackageOpen,
  Play,
  Puzzle,
  Search,
  Shirt,
  ShoppingBag,
  SlidersHorizontal,
  Sparkles,
  Star,
  Table2,
  Tags,
  Target,
  Users,
  WandSparkles
} from "lucide-react";

const API_BASE = "/api/flask";
const EVAL_MAX_USERS = 40;

const navItems = [
  { id: "home", label: "Overview", icon: Activity },
  { id: "cf", label: "Collaborative", icon: Network },
  { id: "cb", label: "Content Based", icon: Fingerprint },
  { id: "kb", label: "Knowledge Based", icon: SlidersHorizontal },
  { id: "eval", label: "Evaluation", icon: BarChart3 }
];

const cfMethods = [
  { value: "user_cosine", label: "User-Based CF (Cosine Similarity)" },
  { value: "user_pearson", label: "User-Based CF (Pearson Correlation)" },
  { value: "item_cosine", label: "Item-Based CF (Cosine Similarity)" },
  { value: "svd", label: "Matrix Factorization (SVD)" }
];

const methodLabels = {
  user_cosine: "User-Based CF (Cosine)",
  user_pearson: "User-Based CF (Pearson)",
  item_cosine: "Item-Based CF (Cosine)",
  svd: "Matrix Factorization (SVD)"
};

const palettes = [
  ["#14b8a6", "#0f766e"],
  ["#ff6b4a", "#d8432e"],
  ["#f7c948", "#a16207"],
  ["#6474ff", "#4338ca"],
  ["#ef476f", "#be123c"],
  ["#22c55e", "#15803d"],
  ["#f97316", "#c2410c"],
  ["#06b6d4", "#0e7490"]
];

export default function Page() {
  const [activePage, setActivePage] = useState("home");
  const [boot, setBoot] = useState(null);
  const [bootError, setBootError] = useState("");
  const [selectedUser, setSelectedUser] = useState("");
  const [userInfo, setUserInfo] = useState(null);
  const [topN, setTopN] = useState(10);
  const [cfMethod, setCfMethod] = useState("user_cosine");
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState("");
  const [errors, setErrors] = useState({});
  const [results, setResults] = useState({ cf: null, cb: null, kb: null });
  const [cfComparison, setCfComparison] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [evalK, setEvalK] = useState(10);
  const [kbFilters, setKbFilters] = useState({
    category: "",
    brand: "",
    min_price: "",
    max_price: "",
    min_rating: 3
  });
  const [availableBrands, setAvailableBrands] = useState([]);

  useEffect(() => {
    let cancelled = false;

    getJson(`${API_BASE}/bootstrap`)
      .then((data) => {
        if (cancelled) return;
        setBoot(data);
        const firstUser = data.user_ids?.[0] ? String(data.user_ids[0]) : "1";
        setSelectedUser(firstUser);
        setKbFilters((current) => ({
          ...current,
          min_price: data.price_range?.min ?? "",
          max_price: data.price_range?.max ?? ""
        }));
      })
      .catch((error) => {
        if (!cancelled) {
          setBootError(error.message || "Unable to load project data.");
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedUser) return;

    getJson(`${API_BASE}/user_info?user_id=${encodeURIComponent(selectedUser)}`)
      .then(setUserInfo)
      .catch(() => setUserInfo(null));
  }, [selectedUser]);

  useEffect(() => {
    let cancelled = false;
    if (!boot) return;

    const cat = kbFilters.category;
    const url = `${API_BASE}/brands${cat ? `?category=${encodeURIComponent(cat)}` : ""}`;

    getJson(url)
      .then((data) => {
        if (cancelled) return;
        const newBrands = data.brands || [];
        setAvailableBrands(newBrands);
        
        // If current selected brand is not in the new list, reset it
        if (kbFilters.brand && !newBrands.includes(kbFilters.brand)) {
          setKbFilters(prev => ({ ...prev, brand: "" }));
        }
      })
      .catch((err) => {
        if (cancelled) return;
        console.error("Failed to fetch brands:", err);
        setAvailableBrands([]);
      });

    return () => {
      cancelled = true;
    };
  }, [kbFilters.category, boot]);

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(""), 2800);
    return () => window.clearTimeout(timer);
  }, [notice]);

  const setPage = useCallback((page) => {
    setActivePage(page);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const boundedTopN = useMemo(() => clampNumber(topN, 3, 20), [topN]);

  async function runTask(section, label, task) {
    setBusy(label);
    setErrors((current) => ({ ...current, [section]: "" }));

    try {
      await task();
      setNotice(label.replace("...", " complete."));
    } catch (error) {
      setErrors((current) => ({
        ...current,
        [section]: error.message || "Something went wrong."
      }));
    } finally {
      setBusy("");
    }
  }

  function getCfRecommendations() {
    if (!selectedUser) return;

    runTask("cf", "Generating collaborative recommendations...", async () => {
      const data = await getJson(
        `${API_BASE}/recommend/cf?${toQuery({
          user_id: selectedUser,
          method: cfMethod,
          top_n: boundedTopN
        })}`
      );

      setResults((current) => ({
        ...current,
        cf: {
          title: methodLabels[cfMethod],
          subtitle: `Ranked for User ${selectedUser}`,
          items: data.recommendations || []
        }
      }));
      setCfComparison(null);
    });
  }

  function compareCfMethods() {
    if (!selectedUser) return;

    runTask("cfComparison", "Comparing collaborative methods...", async () => {
      const data = await getJson(
        `${API_BASE}/compare_cf?${toQuery({ user_id: selectedUser })}`
      );
      setCfComparison(data);
    });
  }

  function getCbRecommendations() {
    if (!selectedUser) return;

    runTask("cb", "Analyzing content profile...", async () => {
      const data = await getJson(
        `${API_BASE}/recommend/cb?${toQuery({
          user_id: selectedUser,
          top_n: boundedTopN
        })}`
      );

      setResults((current) => ({
        ...current,
        cb: {
          title: "Content-based recommendations",
          subtitle: `TF-IDF profile for User ${selectedUser}`,
          items: data.recommendations || []
        }
      }));
    });
  }

  function getKbRecommendations() {
    if (!selectedUser) return;

    runTask("kb", "Filtering product catalog...", async () => {
      const data = await getJson(
        `${API_BASE}/recommend/kb?${toQuery({
          user_id: selectedUser,
          top_n: boundedTopN,
          ...kbFilters
        })}`
      );

      setResults((current) => ({
        ...current,
        kb: {
          title: "Knowledge-based recommendations",
          subtitle: `Constraint ranking for User ${selectedUser}`,
          items: data.recommendations || []
        }
      }));
    });
  }

  function runEvaluation() {
    runTask("eval", "Running full evaluation...", async () => {
      const data = await getJson(
        `${API_BASE}/evaluate?${toQuery({ k: clampNumber(evalK, 3, 20), max_users: EVAL_MAX_USERS })}`
      );
      setEvaluation(data);
    });
  }

  if (bootError) {
    return (
      <main className="fatal-state">
        <AlertTriangle />
        <h1>Flask backend is not reachable</h1>
        <p>{bootError}</p>
        <p>Start the backend with <code>python app.py</code>, then refresh this page.</p>
      </main>
    );
  }

  return (
    <div className="app-shell">
      <SignalCanvas />
      {busy && <BusyOverlay label={busy} />}
      {notice && <Toast message={notice} />}

      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <ShoppingBag size={24} />
          </div>
          <div>
            <strong>MarketMatch</strong>
            <span>AIE425 Recommender</span>
          </div>
        </div>

        <nav className="nav-list" aria-label="Dashboard navigation">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                className={`nav-button ${activePage === item.id ? "active" : ""}`}
                type="button"
                onClick={() => setPage(item.id)}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <section className="control-deck">
          <div className="deck-title">
            <span>User Console</span>
            <Users size={18} />
          </div>

          <label className="field">
            <span>Selected User</span>
            <select
              value={selectedUser}
              onChange={(event) => setSelectedUser(event.target.value)}
              disabled={!boot}
            >
              {(boot?.user_ids || []).map((uid) => (
                <option key={uid} value={uid}>
                  User {uid}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>Top Results</span>
            <input
              type="number"
              min="3"
              max="20"
              value={topN}
              onChange={(event) => setTopN(event.target.value)}
            />
          </label>

          <div className="profile-card">
            <strong>User {selectedUser || "-"}</strong>
            <span>{userInfo ? `${userInfo.n_ratings} ratings` : "Loading profile"}</span>
            <small>{cleanText(userInfo?.prefs || "Preferences loading")}</small>
          </div>
        </section>

        <div className="tech-stack">
          <span>Next.js</span>
          <span>Flask API</span>
          <span>ML Models</span>
        </div>
      </aside>

      <main className="main">
        <section className={`page ${activePage === "home" ? "active" : ""}`}>
          <HomePage boot={boot} setPage={setPage} />
        </section>

        <section className={`page ${activePage === "cf" ? "active" : ""}`}>
          <PageHeader
            tone="teal"
            label="Collaborative filtering"
            title="Recommendations from shared behavior."
            copy="Compare user similarity, item similarity, and SVD ranking from one clean workspace."
          />

          <ToolPanel
            title="Similarity engine"
            copy="Pick a collaborative method and generate a fresh ranking for the active user."
            icon={Network}
          >
            <label className="field wide">
              <span>CF Method</span>
              <select value={cfMethod} onChange={(event) => setCfMethod(event.target.value)}>
                {cfMethods.map((method) => (
                  <option key={method.value} value={method.value}>
                    {method.label}
                  </option>
                ))}
              </select>
            </label>
            <div className="actions">
              <button className="button primary" type="button" onClick={getCfRecommendations}>
                <Sparkles size={18} />
                Get Recommendations
              </button>
              <button className="button gold" type="button" onClick={compareCfMethods}>
                <Table2 size={18} />
                Compare Methods
              </button>
            </div>
          </ToolPanel>

          <ErrorBox message={errors.cf} />
          <ResultBlock view={results.cf} emptyIcon={Search} emptyText="Collaborative results will appear here." />
          <ErrorBox message={errors.cfComparison} />
          <CfComparison data={cfComparison} />
        </section>

        <section className={`page ${activePage === "cb" ? "active" : ""}`}>
          <PageHeader
            tone="coral"
            label="Content based"
            title="Rank products from profile fingerprints."
            copy="TF-IDF blends category, brand, and description signals into a personalized taste vector."
          />

          <ToolPanel
            title="Profile similarity"
            copy="Generate a content-driven set for the active user."
            icon={Fingerprint}
            compact
          >
            <button className="button primary" type="button" onClick={getCbRecommendations}>
              <WandSparkles size={18} />
              Get Recommendations
            </button>
          </ToolPanel>

          <ErrorBox message={errors.cb} />
          <ResultBlock view={results.cb} emptyIcon={Layers3} emptyText="Content-based results will appear here." />
        </section>

        <section className={`page ${activePage === "kb" ? "active" : ""}`}>
          <PageHeader
            tone="gold"
            label="Knowledge based"
            title="Shape the ranking with explicit constraints."
            copy="Turn category, brand, price, and quality requirements into a clear product shortlist."
          />

          <ToolPanel
            title="Constraint builder"
            copy="Filter the catalog first, then rank by rating and popularity."
            icon={SlidersHorizontal}
          >
            <div className="filter-grid">
              <label className="field">
                <span>Category</span>
                <select
                  value={kbFilters.category}
                  onChange={(event) => setKbFilters({ ...kbFilters, category: event.target.value })}
                >
                  <option value="">All Categories</option>
                  {(boot?.categories || []).map((category) => (
                    <option key={category} value={category}>
                      {category}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Brand</span>
                <select
                  value={kbFilters.brand}
                  onChange={(event) => setKbFilters({ ...kbFilters, brand: event.target.value })}
                >
                  <option value="">All Brands</option>
                  {(availableBrands || []).map((brand) => (
                    <option key={brand} value={brand}>
                      {brand}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Min Price</span>
                <input
                  type="number"
                  min={boot?.price_range?.min ?? 0}
                  step="5"
                  value={kbFilters.min_price}
                  onChange={(event) => setKbFilters({ ...kbFilters, min_price: event.target.value })}
                />
              </label>
              <label className="field">
                <span>Max Price</span>
                <input
                  type="number"
                  max={boot?.price_range?.max ?? 1000}
                  step="5"
                  value={kbFilters.max_price}
                  onChange={(event) => setKbFilters({ ...kbFilters, max_price: event.target.value })}
                />
              </label>
              <label className="field">
                <span>Min Rating</span>
                <input
                  type="number"
                  min="1"
                  max="5"
                  step="0.5"
                  value={kbFilters.min_rating}
                  onChange={(event) => setKbFilters({ ...kbFilters, min_rating: event.target.value })}
                />
              </label>
            </div>
            <button className="button primary" type="button" onClick={getKbRecommendations}>
              <Target size={18} />
              Apply Filters
            </button>
          </ToolPanel>

          <ErrorBox message={errors.kb} />
          <ResultBlock view={results.kb} emptyIcon={SlidersHorizontal} emptyText="Knowledge-based results will appear here." />
        </section>

        <section className={`page ${activePage === "eval" ? "active" : ""}`}>
          <PageHeader
            tone="indigo"
            label="Evaluation dashboard"
            title="Compare ranking quality and prediction error."
            copy="Benchmark each recommender with Precision@K, Recall@K, NDCG@K, and RMSE."
          />

          <ToolPanel
            title="Evaluation run"
            copy={`Choose K and benchmark every model on a stable ${EVAL_MAX_USERS}-user test sample.`}
            icon={Gauge}
            compact
          >
            <label className="field compact">
              <span>Top-K</span>
              <input
                type="number"
                min="3"
                max="20"
                value={evalK}
                onChange={(event) => setEvalK(event.target.value)}
              />
            </label>
            <button className="button primary" type="button" onClick={runEvaluation}>
              <Play size={18} />
              Run Evaluation
            </button>
          </ToolPanel>

          <ErrorBox message={errors.eval} />
          <EvaluationBlock data={evaluation} />
        </section>
      </main>
    </div>
  );
}

function HomePage({ boot, setPage }) {
  const stats = boot?.stats || {};

  return (
    <>
      <header className="hero">
        <div className="hero-copy">
          <span className="eyebrow">Intelligent retail system</span>
          <h1>Recommendation studio for e-commerce decisions.</h1>
          <p>
            A full Next.js interface for exploring users, ranking products, and comparing recommendation methods.
          </p>
          <div className="hero-actions">
            <button className="button primary" type="button" onClick={() => setPage("cf")}>
              <Sparkles size={18} />
              Start Matching
            </button>
            <button className="button dark" type="button" onClick={() => setPage("eval")}>
              <ChartNoAxesCombined size={18} />
              View Metrics
            </button>
          </div>
        </div>

        <div className="hero-board" aria-hidden="true">
          <div className="board-top">
            <span></span>
            <span></span>
            <span></span>
            <strong>Live ranking feed</strong>
          </div>
          <div className="ranking-feed">
            <FeedRow icon={Monitor} label="Electronics" score="4.9" tone="teal" delay="0ms" />
            <FeedRow icon={Shirt} label="Fashion" score="4.7" tone="coral" delay="160ms" />
            <FeedRow icon={Home} label="Home Goods" score="4.6" tone="gold" delay="320ms" />
            <div className="signal-bars">
              <span></span>
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        </div>
      </header>

      <div className="stats-grid">
        <StatCard icon={PackageOpen} label="Products" value={stats.products} tone="teal" />
        <StatCard icon={Users} label="Users" value={stats.users} tone="coral" />
        <StatCard icon={Star} label="Ratings" value={stats.ratings} tone="gold" />
        <StatCard icon={Gauge} label="Average Rating" value={stats.avg_rating} tone="indigo" />
      </div>

      <section className="section">
        <SectionHeading eyebrow="Model workspace" title="Three recommendation tracks" />
        <div className="approach-grid">
          <ApproachCard
            icon={Network}
            title="Collaborative Filtering"
            copy="User similarity, item similarity, Pearson, cosine, and SVD."
            meta="4 methods"
            tone="teal"
            onClick={() => setPage("cf")}
          />
          <ApproachCard
            icon={Fingerprint}
            title="Content Based"
            copy="TF-IDF matching across product text, category, and brand."
            meta="Feature profile"
            tone="coral"
            onClick={() => setPage("cb")}
          />
          <ApproachCard
            icon={BadgeDollarSign}
            title="Knowledge Based"
            copy="Category, brand, price, and rating constraints."
            meta="Rule ranking"
            tone="gold"
            onClick={() => setPage("kb")}
          />
        </div>
      </section>

      <section className="section">
        <SectionHeading eyebrow="Dataset pulse" title="Catalog and rating distribution" />
        <div className="chart-grid">
          <BarPanel title="Products per category" labels={boot?.category_chart?.labels} values={boot?.category_chart?.values} />
          <BarPanel title="Rating distribution" labels={boot?.rating_chart?.labels} values={boot?.rating_chart?.values} compact />
        </div>
      </section>
    </>
  );
}

function PageHeader({ label, title, copy, tone }) {
  return (
    <header className={`page-header ${tone}`}>
      <span className="eyebrow">{label}</span>
      <h1>{title}</h1>
      <p>{copy}</p>
    </header>
  );
}

function ToolPanel({ title, copy, icon: Icon, children, compact = false }) {
  return (
    <section className={`tool-panel ${compact ? "compact-panel" : ""}`}>
      <div className="panel-title">
        <div>
          <h2>{title}</h2>
          <p>{copy}</p>
        </div>
        <Icon size={24} />
      </div>
      <div className="panel-controls">{children}</div>
    </section>
  );
}

function StatCard({ icon: Icon, label, value, tone }) {
  return (
    <article className={`stat-card ${tone}`}>
      <Icon size={22} />
      <strong>{value ?? "-"}</strong>
      <span>{label}</span>
    </article>
  );
}

function ApproachCard({ icon: Icon, title, copy, meta, tone, onClick }) {
  return (
    <button className={`approach-card ${tone}`} type="button" onClick={onClick}>
      <span className="approach-icon">
        <Icon size={24} />
      </span>
      <strong>{title}</strong>
      <small>{copy}</small>
      <em>
        {meta}
        <ChevronRight size={16} />
      </em>
    </button>
  );
}

function FeedRow({ icon: Icon, label, score, tone, delay }) {
  return (
    <div className={`feed-row ${tone}`} style={{ "--delay": delay }}>
      <Icon size={22} />
      <span>{label}</span>
      <strong>{score}</strong>
    </div>
  );
}

function SectionHeading({ eyebrow, title }) {
  return (
    <div className="section-heading">
      <span className="eyebrow dark">{eyebrow}</span>
      <h2>{title}</h2>
    </div>
  );
}

function BarPanel({ title, labels = [], values = [], compact = false }) {
  const max = Math.max(...values, 1);

  return (
    <article className={`bar-panel ${compact ? "compact-bars" : ""}`}>
      <h3>{title}</h3>
      <div className="bars" role="img" aria-label={title}>
        {labels.map((label, index) => {
          const value = Number(values[index] || 0);
          const height = Math.max(8, (value / max) * 100);
          const palette = palettes[index % palettes.length];

          return (
            <div className="bar-item" key={label}>
              <div className="bar-track">
                <span
                  style={{
                    "--height": `${height}%`,
                    "--tone-a": palette[0],
                    "--tone-b": palette[1],
                    "--delay": `${index * 70}ms`
                  }}
                ></span>
              </div>
              <strong>{value.toLocaleString()}</strong>
              <small>{cleanText(label)}</small>
            </div>
          );
        })}
      </div>
    </article>
  );
}

function ResultBlock({ view, emptyIcon: EmptyIcon, emptyText }) {
  if (!view) {
    return (
      <div className="empty-state">
        <EmptyIcon size={22} />
        <span>{emptyText}</span>
      </div>
    );
  }

  const items = view.items || [];

  return (
    <section className="result-block">
      <div className="result-heading">
        <div>
          <h2>{cleanText(view.title)}</h2>
          <p>{cleanText(view.subtitle)}</p>
        </div>
        <span>{items.length} products</span>
      </div>
      {items.length ? (
        <div className="product-grid">
          {items.map((item, index) => (
            <ProductCard item={item} index={index} key={`${item.product_id}-${index}`} />
          ))}
        </div>
      ) : (
        <div className="empty-state">
          <AlertTriangle size={22} />
          <span>No recommendations found for this configuration.</span>
        </div>
      )}
    </section>
  );
}

function ProductCard({ item, index }) {
  const category = cleanText(item.category || "Product");
  const Icon = categoryIcon(category);
  const palette = paletteFor(category);
  const rating = Number(item.predicted_rating || 0);
  const score = Math.max(2, Math.min(100, (rating / 5) * 100));

  return (
    <article
      className="product-card"
      style={{
        "--tone-a": palette[0],
        "--tone-b": palette[1],
        "--delay": `${index * 60}ms`,
        "--score": `${score}%`
      }}
    >
      <div className="product-icon">
        <Icon size={24} />
      </div>
      <div className="product-content">
        <div className="product-title">
          <h3>{cleanText(item.product_name || "Unnamed product")}</h3>
          <span>#{index + 1}</span>
        </div>
        <div className="product-meta">
          <span>
            <Tags size={14} />
            {category}
          </span>
          <span>
            <Building2 size={14} />
            {cleanText(item.brand || "Brand")}
          </span>
          <span>
            <CircleDollarSign size={14} />
            {formatCurrency(item.price)}
          </span>
        </div>
        <div className="score-line">
          <strong>{rating.toFixed(2)} / 5</strong>
          <div>
            <span></span>
          </div>
        </div>
        <p>{cleanText(item.explanation || "Recommended by the selected model.")}</p>
      </div>
    </article>
  );
}

function CfComparison({ data }) {
  if (!data) return null;

  return (
    <section className="comparison-block">
      <div className="result-heading">
        <div>
          <h2>Collaborative method comparison</h2>
          <p>Top 5 products from every collaborative strategy.</p>
        </div>
        <span>{Object.keys(data).length} methods</span>
      </div>
      <div className="method-grid">
        {Object.entries(data).map(([method, rows]) => (
          <article className="method-card" key={method}>
            <h3>{cleanText(method)}</h3>
            <Table
              columns={["Product", "Category", "Rating"]}
              rows={(rows || []).map((row) => [
                cleanText(row.product_name),
                cleanText(row.category),
                formatNumber(row.predicted_rating, 2)
              ])}
            />
          </article>
        ))}
      </div>
    </section>
  );
}

function EvaluationBlock({ data }) {
  if (!data) {
    return (
      <div className="empty-state">
        <BarChart3 size={22} />
        <span>Evaluation output will appear here.</span>
      </div>
    );
  }

  const rows = data.results || [];

  return (
    <section className="evaluation-block">
      <div className="metric-grid">
        {["Precision@K", "Recall@K", "NDCG@K", "RMSE"].map((metric) => {
          const best = bestMetric(rows, metric);
          return (
            <article className="metric-card" key={metric}>
              <span>Best {metric}</span>
              <strong>{best.value === null ? "N/A" : formatNumber(best.value, 4)}</strong>
              <small>{cleanText(best.method)}</small>
            </article>
          );
        })}
      </div>

      <div className="result-heading">
        <div>
          <h2>Metric table</h2>
          <p>
            Higher is better for ranking metrics. Lower is better for RMSE.
            {data.sample_size ? ` Sample: ${data.sample_size} of ${data.total_test_users} test users.` : ""}
          </p>
        </div>
        <span>{rows.length} rows</span>
      </div>

      <Table
        columns={["Method", "Precision@K", "Recall@K", "NDCG@K", "RMSE", "Users"]}
        rows={rows.map((row) => [
          cleanText(row.Method),
          formatMaybe(row["Precision@K"], 4),
          formatMaybe(row["Recall@K"], 4),
          formatMaybe(row["NDCG@K"], 4),
          formatMaybe(row.RMSE, 4),
          row.num_users_evaluated || "N/A"
        ])}
      />

      <div className="chart-grid">
        <MetricBars
          title="Ranking metrics"
          rows={rows}
          metrics={[
            ["Precision@K", "#14b8a6"],
            ["Recall@K", "#ff6b4a"],
            ["NDCG@K", "#6474ff"]
          ]}
        />
        <MetricBars title="RMSE" rows={rows} metrics={[["RMSE", "#ef476f"]]} />
      </div>

      <div className="analysis-panel">
        <h2>Detailed analysis</h2>
        <p>{cleanText(data.analysis || "No analysis returned.")}</p>
      </div>

      <Table
        title="Approach comparison"
        columns={["Aspect", "Collaborative", "Content Based", "Knowledge Based"]}
        rows={[
          ["Cold start", "Needs rating history", "Works with item features", "Works with constraints"],
          ["Diversity", "Often broad", "Can stay close to past taste", "Depends on filters"],
          ["Explainability", "Similarity based", "Feature based", "Constraint based"],
          ["Best fit", "Dense rating data", "Rich product text", "Clear shopping needs"]
        ]}
      />
    </section>
  );
}

function MetricBars({ title, rows, metrics }) {
  const values = rows.flatMap((row) => metrics.map(([metric]) => Number(row[metric]) || 0));
  const max = Math.max(...values, 1);

  return (
    <article className="metric-bars">
      <h3>{title}</h3>
      <div className="metric-bar-list">
        {rows.map((row) => (
          <div className="metric-row" key={row.Method}>
            <span>{cleanText(row.Method)}</span>
            <div className="metric-stack">
              {metrics.map(([metric, color]) => {
                const value = Number(row[metric]) || 0;
                const width = Math.max(2, (value / max) * 100);
                return (
                  <em key={metric} style={{ "--width": `${width}%`, "--color": color }}>
                    {formatNumber(value, metric === "RMSE" ? 3 : 4)}
                  </em>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </article>
  );
}

function Table({ title, columns, rows }) {
  return (
    <div className="table-shell">
      {title && <h2>{title}</h2>}
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {row.map((cell, cellIndex) => (
                  <td key={`${rowIndex}-${cellIndex}`}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ErrorBox({ message }) {
  if (!message) return null;
  return (
    <div className="error-box">
      <AlertTriangle size={20} />
      <span>{message}</span>
    </div>
  );
}

function BusyOverlay({ label }) {
  return (
    <div className="busy-overlay" role="status" aria-live="polite">
      <div>
        <Loader2 className="spin" size={34} />
        <strong>{label}</strong>
      </div>
    </div>
  );
}

function Toast({ message }) {
  return (
    <div className="toast">
      <Sparkles size={18} />
      <span>{message}</span>
    </div>
  );
}

function SignalCanvas() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const ctx = canvas.getContext("2d");
    const colors = [
      [20, 184, 166],
      [255, 107, 74],
      [247, 201, 72],
      [100, 116, 255]
    ];
    let points = [];
    let frame = 0;

    function resize() {
      const ratio = window.devicePixelRatio || 1;
      canvas.width = window.innerWidth * ratio;
      canvas.height = window.innerHeight * ratio;
      canvas.style.width = `${window.innerWidth}px`;
      canvas.style.height = `${window.innerHeight}px`;
      points = Array.from({ length: Math.min(70, Math.floor(window.innerWidth / 18)) }, (_, index) => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.28 * ratio,
        vy: (Math.random() - 0.5) * 0.28 * ratio,
        color: colors[index % colors.length]
      }));
    }

    function draw() {
      const ratio = window.devicePixelRatio || 1;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      points.forEach((point, index) => {
        point.x += point.vx;
        point.y += point.vy;

        if (point.x < 0 || point.x > canvas.width) point.vx *= -1;
        if (point.y < 0 || point.y > canvas.height) point.vy *= -1;

        const [r, g, b] = point.color;
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.34)`;
        ctx.fillRect(point.x, point.y, 3 * ratio, 3 * ratio);

        for (let next = index + 1; next < points.length; next += 1) {
          const other = points[next];
          const dx = point.x - other.x;
          const dy = point.y - other.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          const limit = 145 * ratio;

          if (distance < limit) {
            ctx.beginPath();
            ctx.moveTo(point.x, point.y);
            ctx.lineTo(other.x, other.y);
            ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${0.08 * (1 - distance / limit)})`;
            ctx.lineWidth = ratio;
            ctx.stroke();
          }
        }
      });

      frame = window.requestAnimationFrame(draw);
    }

    resize();
    draw();
    window.addEventListener("resize", resize);

    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return <canvas className="signal-canvas" ref={canvasRef} aria-hidden="true" />;
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }
  return response.json();
}

function toQuery(params) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== "" && value !== null && value !== undefined) {
      search.set(key, value);
    }
  });
  return search.toString();
}

function clampNumber(value, min, max) {
  const number = Number(value);
  if (Number.isNaN(number)) return min;
  return Math.min(max, Math.max(min, number));
}

function cleanText(value) {
  let text = String(value ?? "");
  const replacements = [
    [/\u00e2\u20ac\u201d/g, " - "],
    [/\u00e2\u20ac\u201c/g, " - "],
    [/\u00e2\u2030\u00a5/g, ">="],
    [/\u00e2\u2030\u00a4/g, "<="],
    [/\u00c2\u00b7/g, " - "],
    [/\u00c2/g, ""]
  ];

  replacements.forEach(([pattern, replacement]) => {
    text = text.replace(pattern, replacement);
  });

  return text
    .replace(/\u00f0\u0178[\u0080-\uffff]{0,4}/g, "")
    .replace(/[\u0080-\uffff]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function paletteFor(text) {
  const cleaned = cleanText(text);
  let hash = 0;
  for (let index = 0; index < cleaned.length; index += 1) {
    hash = (hash + cleaned.charCodeAt(index) * (index + 1)) % palettes.length;
  }
  return palettes[hash];
}

function categoryIcon(category) {
  const value = cleanText(category).toLowerCase();
  if (value.includes("electronics")) return Monitor;
  if (value.includes("clothing")) return Shirt;
  if (value.includes("home")) return Home;
  if (value.includes("books")) return BookOpen;
  if (value.includes("sports")) return Dumbbell;
  if (value.includes("beauty")) return HeartPulse;
  if (value.includes("toys") || value.includes("games")) return Puzzle;
  if (value.includes("automotive")) return Car;
  return Boxes;
}

function formatCurrency(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0
  }).format(Number(value || 0));
}

function formatNumber(value, digits = 2) {
  return Number(value || 0).toFixed(digits);
}

function formatMaybe(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
  return Number(value).toFixed(digits);
}

function bestMetric(rows, metric) {
  const valid = rows.filter((row) => row[metric] !== null && row[metric] !== undefined && !Number.isNaN(Number(row[metric])));
  if (!valid.length) return { value: null, method: "" };

  const sorted = [...valid].sort((a, b) => {
    const left = Number(a[metric]);
    const right = Number(b[metric]);
    return metric === "RMSE" ? left - right : right - left;
  });

  return { value: Number(sorted[0][metric]), method: sorted[0].Method };
}
