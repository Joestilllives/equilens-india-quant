"""SQLite persistence layer for OHLCV, macro regime, earnings, insiders, options, ML, journal, and alerts."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import DB_PATH, NON_EQUITY_SYMBOLS


class Database:
    """Thread-safe WAL SQLite store for EquiLens v2 quantitative pipeline."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=15000;")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tickers (
                    symbol TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    sector TEXT NOT NULL DEFAULT 'Unknown',
                    industry TEXT NOT NULL DEFAULT '',
                    exchange TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    fundamentals_json TEXT NOT NULL DEFAULT '{}',
                    data_source TEXT NOT NULL DEFAULT 'yfinance',
                    starred INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ohlcv (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    PRIMARY KEY (symbol, date)
                );

                CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_date
                    ON ohlcv (symbol, date);

                CREATE TABLE IF NOT EXISTS macro_series (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    close REAL NOT NULL,
                    PRIMARY KEY (symbol, date)
                );

                CREATE TABLE IF NOT EXISTS news (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    publisher TEXT NOT NULL DEFAULT '',
                    url TEXT NOT NULL DEFAULT '',
                    published_at TEXT NOT NULL DEFAULT '',
                    sentiment_score REAL NOT NULL DEFAULT 0.0,
                    sentiment_label TEXT NOT NULL DEFAULT 'Neutral',
                    catalyst_tags_json TEXT NOT NULL DEFAULT '[]'
                );

                CREATE INDEX IF NOT EXISTS idx_news_symbol_pub
                    ON news (symbol, published_at DESC);

                CREATE TABLE IF NOT EXISTS earnings_history (
                    symbol TEXT NOT NULL,
                    earnings_date TEXT NOT NULL,
                    eps_estimate REAL,
                    eps_actual REAL,
                    surprise_pct REAL,
                    PRIMARY KEY (symbol, earnings_date)
                );

                CREATE TABLE IF NOT EXISTS insider_trades (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    insider TEXT NOT NULL,
                    position TEXT NOT NULL DEFAULT '',
                    transaction_type TEXT NOT NULL DEFAULT '',
                    shares REAL NOT NULL DEFAULT 0,
                    value REAL NOT NULL DEFAULT 0,
                    is_purchase INTEGER NOT NULL DEFAULT 0,
                    source TEXT NOT NULL DEFAULT 'yfinance'
                );

                CREATE INDEX IF NOT EXISTS idx_insider_symbol_date
                    ON insider_trades (symbol, trade_date DESC);

                CREATE TABLE IF NOT EXISTS options_snapshot (
                    symbol TEXT PRIMARY KEY,
                    as_of_date TEXT NOT NULL,
                    expiry TEXT NOT NULL DEFAULT '',
                    call_volume INTEGER NOT NULL DEFAULT 0,
                    put_volume INTEGER NOT NULL DEFAULT 0,
                    call_oi INTEGER NOT NULL DEFAULT 0,
                    put_oi INTEGER NOT NULL DEFAULT 0,
                    pc_vol_ratio REAL,
                    pc_oi_ratio REAL,
                    atm_iv_pct REAL,
                    iv_hv_ratio REAL,
                    options_sentiment REAL NOT NULL DEFAULT 0.0
                );

                CREATE TABLE IF NOT EXISTS ml_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trained_at TEXT NOT NULL,
                    horizon_days INTEGER NOT NULL,
                    sample_count INTEGER NOT NULL,
                    symbol_count INTEGER NOT NULL,
                    hit_rate REAL NOT NULL,
                    ic_spearman REAL NOT NULL,
                    top5_spread REAL NOT NULL,
                    avg_top5_return REAL NOT NULL,
                    avg_bench_return REAL NOT NULL,
                    tb_auc REAL NOT NULL DEFAULT 0.5,
                    tb_top5_precision REAL NOT NULL DEFAULT 50.0,
                    feature_importances_json TEXT NOT NULL,
                    folds_json TEXT NOT NULL,
                    regime_stats_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS ml_predictions (
                    symbol TEXT PRIMARY KEY,
                    as_of_date TEXT NOT NULL,
                    pred_alpha_20d REAL NOT NULL,
                    prob_outperform REAL NOT NULL,
                    prob_tb_win REAL NOT NULL DEFAULT 50.0,
                    expected_return_20d REAL NOT NULL,
                    vol_20d_ann REAL NOT NULL,
                    target_price REAL,
                    stop_price REAL,
                    risk_reward REAL,
                    cone_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ml_oos_history (
                    date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    pred_alpha REAL NOT NULL,
                    prob_tb_win REAL NOT NULL,
                    actual_fwd_20d REAL,
                    bench_fwd_20d REAL,
                    tb_outcome INTEGER,
                    PRIMARY KEY (date, symbol)
                );

                CREATE INDEX IF NOT EXISTS idx_oos_date ON ml_oos_history (date);

                CREATE TABLE IF NOT EXISTS prediction_journal (
                    id TEXT PRIMARY KEY,
                    logged_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    composite_score REAL NOT NULL,
                    ml_tb_prob REAL NOT NULL,
                    ml_alpha_20d REAL NOT NULL,
                    target_price REAL,
                    stop_price REAL,
                    conviction TEXT NOT NULL,
                    regime TEXT NOT NULL DEFAULT 'Normal',
                    is_live INTEGER NOT NULL DEFAULT 1,
                    ret_5d REAL,
                    ret_20d REAL,
                    ret_60d REAL,
                    spy_ret_20d REAL,
                    outcome TEXT NOT NULL DEFAULT 'Open',
                    graded_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_journal_date ON prediction_journal (logged_date DESC);

                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    acknowledged INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS paper_account (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    started_at TEXT NOT NULL,
                    last_rebalance_at TEXT NOT NULL,
                    initial_capital REAL NOT NULL,
                    cash_inr REAL NOT NULL,
                    realized_pnl_inr REAL NOT NULL DEFAULT 0.0,
                    total_fees_inr REAL NOT NULL DEFAULT 0.0,
                    est_stcg_tax_inr REAL NOT NULL DEFAULT 0.0,
                    bench_entry_price REAL NOT NULL,
                    risk_profile TEXT NOT NULL DEFAULT 'balanced',
                    top_k INTEGER NOT NULL DEFAULT 10
                );

                CREATE TABLE IF NOT EXISTS paper_positions (
                    symbol TEXT PRIMARY KEY,
                    sector TEXT NOT NULL DEFAULT 'Unknown',
                    shares INTEGER NOT NULL,
                    avg_cost REAL NOT NULL,
                    entry_date TEXT NOT NULL,
                    target_price REAL,
                    stop_price REAL
                );

                CREATE TABLE IF NOT EXISTS paper_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    executed_at TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    shares INTEGER NOT NULL,
                    price REAL NOT NULL,
                    gross_value REAL NOT NULL,
                    fee_inr REAL NOT NULL DEFAULT 0.0,
                    realized_pnl_inr REAL NOT NULL DEFAULT 0.0,
                    reason TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS nse_delivery (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    close_price REAL NOT NULL DEFAULT 0.0,
                    traded_qty INTEGER NOT NULL DEFAULT 0,
                    deliv_qty INTEGER NOT NULL DEFAULT 0,
                    deliv_pct REAL NOT NULL DEFAULT 0.0,
                    turnover_lacs REAL NOT NULL DEFAULT 0.0,
                    no_of_trades INTEGER NOT NULL DEFAULT 0,
                    avg_trade_val_inr REAL NOT NULL DEFAULT 0.0,
                    PRIMARY KEY (symbol, date)
                );

                CREATE INDEX IF NOT EXISTS idx_nse_delivery_date ON nse_delivery(date DESC);

                CREATE TABLE IF NOT EXISTS nse_fii_oi (
                    date TEXT PRIMARY KEY,
                    fii_idx_fut_long INTEGER NOT NULL DEFAULT 0,
                    fii_idx_fut_short INTEGER NOT NULL DEFAULT 0,
                    fii_idx_long_pct REAL NOT NULL DEFAULT 50.0,
                    fii_net_idx_fut INTEGER NOT NULL DEFAULT 0,
                    dii_net_idx_fut INTEGER NOT NULL DEFAULT 0,
                    client_net_idx_fut INTEGER NOT NULL DEFAULT 0,
                    fii_stk_fut_net INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS nse_bulk_deals (
                    id TEXT PRIMARY KEY,
                    date TEXT NOT NULL,
                    deal_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    client_name TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL DEFAULT 0,
                    price REAL NOT NULL DEFAULT 0.0,
                    value_cr REAL NOT NULL DEFAULT 0.0
                );

                CREATE TABLE IF NOT EXISTS paper_lots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    buy_date TEXT NOT NULL,
                    shares_remaining REAL NOT NULL,
                    unit_cost_inr REAL NOT NULL,
                    sleeve_id INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_paper_lots_sym ON paper_lots (symbol, buy_date);

                CREATE TABLE IF NOT EXISTS paper_nav_history (
                    date TEXT PRIMARY KEY,
                    nav_inr REAL NOT NULL,
                    equity_inr REAL NOT NULL,
                    liquidcase_inr REAL NOT NULL,
                    goldbees_inr REAL NOT NULL,
                    peak_nav_inr REAL NOT NULL,
                    dd_pct REAL NOT NULL DEFAULT 0.0,
                    dd_brake_active INTEGER NOT NULL DEFAULT 0,
                    bench_tri_price REAL
                );

                CREATE TABLE IF NOT EXISTS paper_sleeves (
                    sleeve_id INTEGER PRIMARY KEY,
                    last_rebalance_date TEXT,
                    next_rebalance_date TEXT,
                    status TEXT NOT NULL DEFAULT 'scheduled',
                    target_weight_pct REAL NOT NULL DEFAULT 25.0
                );

                CREATE TABLE IF NOT EXISTS pit_fundamentals_history (
                    symbol TEXT NOT NULL,
                    as_of_date TEXT NOT NULL,
                    filing_date TEXT,
                    trailing_pe REAL,
                    ev_to_ebitda REAL,
                    fcf_yield REAL,
                    gross_profit_to_assets REAL,
                    accruals REAL,
                    roe REAL,
                    debt_to_equity REAL,
                    market_cap_cr REAL,
                    source TEXT NOT NULL DEFAULT 'snapshot',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (symbol, as_of_date)
                );

                CREATE INDEX IF NOT EXISTS idx_pit_fund_date ON pit_fundamentals_history (as_of_date DESC, symbol);
                """
            )
            # Safe schema migrations if upgrading from v1 DB
            self._ensure_column(conn, "news", "catalyst_tags_json", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(conn, "ml_runs", "tb_auc", "REAL NOT NULL DEFAULT 0.5")
            self._ensure_column(conn, "ml_runs", "tb_top5_precision", "REAL NOT NULL DEFAULT 50.0")
            self._ensure_column(conn, "ml_runs", "regime_stats_json", "TEXT NOT NULL DEFAULT '{}'")
            self._ensure_column(conn, "ml_predictions", "prob_tb_win", "REAL NOT NULL DEFAULT 50.0")
            self._ensure_column(conn, "ml_predictions", "target_price", "REAL")
            self._ensure_column(conn, "ml_predictions", "stop_price", "REAL")
            self._ensure_column(conn, "ml_predictions", "risk_reward", "REAL")
            self._ensure_column(conn, "paper_account", "peak_nav_inr", "REAL")
            self._ensure_column(conn, "paper_account", "dd_brake_active", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "paper_account", "liquidcase_inr", "REAL")
            self._ensure_column(conn, "paper_account", "goldbees_inr", "REAL NOT NULL DEFAULT 0.0")
            self._ensure_column(conn, "paper_account", "bench_tri_entry_price", "REAL")
            self._ensure_column(conn, "paper_account", "est_ltcg_tax_inr", "REAL NOT NULL DEFAULT 0.0")
            self._ensure_column(conn, "paper_account", "stcl_carry_inr", "REAL NOT NULL DEFAULT 0.0")
            self._ensure_column(conn, "paper_account", "ltcl_carry_inr", "REAL NOT NULL DEFAULT 0.0")
            self._ensure_column(conn, "paper_nav_history", "bench_tri_price", "REAL")

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, col: str, definition: str) -> None:
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {definition}")

    # ------------------------------------------------------------------
    # Tickers & OHLCV
    # ------------------------------------------------------------------
    def upsert_ticker(
        self,
        symbol: str,
        name: str,
        sector: str,
        industry: str,
        exchange: str,
        summary: str,
        fundamentals: dict[str, Any],
        data_source: str,
        updated_at: str,
        default_starred: bool = False,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO tickers (
                    symbol, name, sector, industry, exchange, summary,
                    fundamentals_json, data_source, starred, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    name = excluded.name,
                    sector = excluded.sector,
                    industry = excluded.industry,
                    exchange = excluded.exchange,
                    summary = excluded.summary,
                    fundamentals_json = excluded.fundamentals_json,
                    data_source = excluded.data_source,
                    updated_at = excluded.updated_at
                """,
                (
                    symbol.upper(),
                    name,
                    sector,
                    industry,
                    exchange,
                    summary,
                    json.dumps(fundamentals),
                    data_source,
                    1 if default_starred else 0,
                    updated_at,
                ),
            )

    def upsert_ohlcv(self, symbol: str, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        sym = symbol.upper()
        tuples = [
            (
                sym,
                r["date"],
                float(r["open"]),
                float(r["high"]),
                float(r["low"]),
                float(r["close"]),
                int(r["volume"]),
            )
            for r in rows
        ]
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO ohlcv (symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, date) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    volume = excluded.volume
                """,
                tuples,
            )
        return len(tuples)

    def upsert_macro_series(self, symbol: str, rows: list[tuple[str, float]]) -> int:
        if not rows:
            return 0
        sym = symbol.upper()
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO macro_series (symbol, date, close)
                VALUES (?, ?, ?)
                ON CONFLICT(symbol, date) DO UPDATE SET close = excluded.close
                """,
                [(sym, d, float(c)) for d, c in rows],
            )
        return len(rows)

    def get_macro_series(self) -> dict[str, dict[str, float]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT symbol, date, close FROM macro_series ORDER BY date ASC").fetchall()
        out: dict[str, dict[str, float]] = {}
        for r in rows:
            out.setdefault(r["symbol"], {})[r["date"]] = float(r["close"])
        return out

    # ------------------------------------------------------------------
    # News, Earnings, Insiders, Options
    # ------------------------------------------------------------------
    def replace_news(self, symbol: str, items: list[dict[str, Any]]) -> None:
        sym = symbol.upper()
        with self.connect() as conn:
            conn.execute("DELETE FROM news WHERE symbol = ?", (sym,))
            if not items:
                return
            conn.executemany(
                """
                INSERT OR REPLACE INTO news (
                    id, symbol, title, summary, publisher, url,
                    published_at, sentiment_score, sentiment_label, catalyst_tags_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(it["id"]),
                        sym,
                        it["title"],
                        it.get("summary", ""),
                        it.get("publisher", ""),
                        it.get("url", ""),
                        it.get("published_at", ""),
                        float(it.get("sentiment_score", 0.0)),
                        it.get("sentiment_label", "Neutral"),
                        json.dumps(it.get("catalyst_tags", [])),
                    )
                    for it in items
                ],
            )

    def replace_earnings(self, symbol: str, items: list[dict[str, Any]]) -> None:
        sym = symbol.upper()
        with self.connect() as conn:
            conn.execute("DELETE FROM earnings_history WHERE symbol = ?", (sym,))
            if not items:
                return
            conn.executemany(
                """
                INSERT OR REPLACE INTO earnings_history (
                    symbol, earnings_date, eps_estimate, eps_actual, surprise_pct
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        sym,
                        it["earnings_date"],
                        it.get("eps_estimate"),
                        it.get("eps_actual"),
                        it.get("surprise_pct"),
                    )
                    for it in items
                ],
            )

    def replace_insiders(self, symbol: str, items: list[dict[str, Any]]) -> None:
        sym = symbol.upper()
        with self.connect() as conn:
            conn.execute("DELETE FROM insider_trades WHERE symbol = ?", (sym,))
            if not items:
                return
            conn.executemany(
                """
                INSERT OR REPLACE INTO insider_trades (
                    id, symbol, trade_date, insider, position,
                    transaction_type, shares, value, is_purchase, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(it["id"]),
                        sym,
                        it["trade_date"],
                        it["insider"],
                        it.get("position", ""),
                        it.get("transaction_type", ""),
                        float(it.get("shares", 0.0)),
                        float(it.get("value", 0.0)),
                        1 if it.get("is_purchase") else 0,
                        it.get("source", "yfinance"),
                    )
                    for it in items
                ],
            )

    def upsert_options_snapshot(self, symbol: str, snap: dict[str, Any] | None) -> None:
        if not snap:
            return
        sym = symbol.upper()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO options_snapshot (
                    symbol, as_of_date, expiry, call_volume, put_volume,
                    call_oi, put_oi, pc_vol_ratio, pc_oi_ratio,
                    atm_iv_pct, iv_hv_ratio, options_sentiment
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    as_of_date = excluded.as_of_date,
                    expiry = excluded.expiry,
                    call_volume = excluded.call_volume,
                    put_volume = excluded.put_volume,
                    call_oi = excluded.call_oi,
                    put_oi = excluded.put_oi,
                    pc_vol_ratio = excluded.pc_vol_ratio,
                    pc_oi_ratio = excluded.pc_oi_ratio,
                    atm_iv_pct = excluded.atm_iv_pct,
                    iv_hv_ratio = excluded.iv_hv_ratio,
                    options_sentiment = excluded.options_sentiment
                """,
                (
                    sym,
                    snap["as_of_date"],
                    snap.get("expiry", ""),
                    int(snap.get("call_volume", 0)),
                    int(snap.get("put_volume", 0)),
                    int(snap.get("call_oi", 0)),
                    int(snap.get("put_oi", 0)),
                    snap.get("pc_vol_ratio"),
                    snap.get("pc_oi_ratio"),
                    snap.get("atm_iv_pct"),
                    snap.get("iv_hv_ratio"),
                    float(snap.get("options_sentiment", 0.0)),
                ),
            )

    # ------------------------------------------------------------------
    # Read Helpers
    # ------------------------------------------------------------------
    def set_starred(self, symbol: str, starred: bool) -> bool:
        with self.connect() as conn:
            cur = conn.execute("UPDATE tickers SET starred = ? WHERE symbol = ?", (1 if starred else 0, symbol.upper()))
            return cur.rowcount > 0

    def get_tickers(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM tickers ORDER BY symbol ASC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["fundamentals"] = json.loads(d.pop("fundamentals_json") or "{}")
            d["starred"] = bool(d["starred"])
            out.append(d)
        return out

    def get_ticker(self, symbol: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            r = conn.execute("SELECT * FROM tickers WHERE symbol = ?", (symbol.upper(),)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["fundamentals"] = json.loads(d.pop("fundamentals_json") or "{}")
        d["starred"] = bool(d["starred"])
        return d

    def get_ohlcv(self, symbol: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT date, open, high, low, close, volume FROM ohlcv WHERE symbol = ? ORDER BY date ASC",
                (symbol.upper(),),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_ohlcv(self) -> dict[str, list[dict[str, Any]]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT symbol, date, open, high, low, close, volume FROM ohlcv ORDER BY symbol ASC, date ASC"
            ).fetchall()
        grouped: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            grouped.setdefault(r["symbol"], []).append(
                {
                    "date": r["date"],
                    "open": r["open"],
                    "high": r["high"],
                    "low": r["low"],
                    "close": r["close"],
                    "volume": r["volume"],
                }
            )
        return grouped

    def get_news(self, symbol: str, limit: int = 12) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, symbol, title, summary, publisher, url, published_at,
                       sentiment_score, sentiment_label, catalyst_tags_json
                FROM news WHERE symbol = ?
                ORDER BY published_at DESC
                LIMIT ?
                """,
                (symbol.upper(), limit),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["catalyst_tags"] = json.loads(d.pop("catalyst_tags_json") or "[]")
            out.append(d)
        return out

    def get_all_news_sentiment(self) -> dict[str, dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT symbol, sentiment_score, published_at, catalyst_tags_json
                FROM news
                ORDER BY symbol ASC, published_at DESC
                """
            ).fetchall()
        grouped: dict[str, list[sqlite3.Row]] = {}
        for r in rows:
            grouped.setdefault(r["symbol"], []).append(r)

        out: dict[str, dict[str, Any]] = {}
        for sym, items in grouped.items():
            # Recency-weighted sentiment (most recent headlines weighted higher)
            w_sum = 0.0
            s_sum = 0.0
            tags: list[str] = []
            for idx, it in enumerate(items[:10]):
                w = 0.85 ** idx
                s_sum += float(it["sentiment_score"] or 0.0) * w
                w_sum += w
                for tg in json.loads(it["catalyst_tags_json"] or "[]"):
                    if tg not in tags:
                        tags.append(tg)
            out[sym] = {
                "avg_sentiment": round(s_sum / max(1e-9, w_sum), 3),
                "article_count": len(items),
                "catalyst_tags": tags[:4],
            }
        return out

    def get_earnings(self, symbol: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT earnings_date, eps_estimate, eps_actual, surprise_pct FROM earnings_history WHERE symbol = ? ORDER BY earnings_date DESC",
                (symbol.upper(),),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_earnings(self) -> dict[str, list[dict[str, Any]]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT symbol, earnings_date, eps_estimate, eps_actual, surprise_pct FROM earnings_history ORDER BY symbol ASC, earnings_date DESC"
            ).fetchall()
        grouped: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            grouped.setdefault(r["symbol"], []).append(dict(r))
        return grouped

    def get_insiders(self, symbol: str, limit: int = 15) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, symbol, trade_date, insider, position, transaction_type,
                       shares, value, is_purchase, source
                FROM insider_trades WHERE symbol = ?
                ORDER BY trade_date DESC LIMIT ?
                """,
                (symbol.upper(), limit),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["is_purchase"] = bool(d["is_purchase"])
            out.append(d)
        return out

    def get_all_insiders(self) -> dict[str, list[dict[str, Any]]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT symbol, trade_date, insider, position, transaction_type,
                       shares, value, is_purchase, source
                FROM insider_trades ORDER BY symbol ASC, trade_date DESC
                """
            ).fetchall()
        grouped: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            d = dict(r)
            d["is_purchase"] = bool(d["is_purchase"])
            grouped.setdefault(r["symbol"], []).append(d)
        return grouped

    def get_options_snapshot(self, symbol: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            r = conn.execute("SELECT * FROM options_snapshot WHERE symbol = ?", (symbol.upper(),)).fetchone()
        return dict(r) if r else None

    def get_all_options_snapshots(self) -> dict[str, dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM options_snapshot").fetchall()
        return {r["symbol"]: dict(r) for r in rows}

    # ------------------------------------------------------------------
    # ML Runs, Predictions & Out-of-Sample History
    # ------------------------------------------------------------------
    def save_ml_run(
        self,
        run: dict[str, Any],
        predictions: list[dict[str, Any]],
        oos_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ml_runs (
                    trained_at, horizon_days, sample_count, symbol_count,
                    hit_rate, ic_spearman, top5_spread, avg_top5_return,
                    avg_bench_return, tb_auc, tb_top5_precision,
                    feature_importances_json, folds_json, regime_stats_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run["trained_at"],
                    int(run["horizon_days"]),
                    int(run["sample_count"]),
                    int(run["symbol_count"]),
                    float(run["hit_rate"]),
                    float(run["ic_spearman"]),
                    float(run["top5_spread"]),
                    float(run["avg_top5_return"]),
                    float(run["avg_bench_return"]),
                    float(run.get("tb_auc", 0.5)),
                    float(run.get("tb_top5_precision", 50.0)),
                    json.dumps(run["feature_importances"]),
                    json.dumps(run["folds"]),
                    json.dumps(run.get("regime_stats", {})),
                ),
            )
            if NON_EQUITY_SYMBOLS:
                placeholders = ",".join("?" for _ in NON_EQUITY_SYMBOLS)
                conn.execute(
                    f"DELETE FROM ml_predictions WHERE symbol IN ({placeholders})",
                    tuple(s.upper() for s in NON_EQUITY_SYMBOLS),
                )
            for p in predictions:
                conn.execute(
                    """
                    INSERT INTO ml_predictions (
                        symbol, as_of_date, pred_alpha_20d, prob_outperform,
                        prob_tb_win, expected_return_20d, vol_20d_ann,
                        target_price, stop_price, risk_reward, cone_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        as_of_date = excluded.as_of_date,
                        pred_alpha_20d = excluded.pred_alpha_20d,
                        prob_outperform = excluded.prob_outperform,
                        prob_tb_win = excluded.prob_tb_win,
                        expected_return_20d = excluded.expected_return_20d,
                        vol_20d_ann = excluded.vol_20d_ann,
                        target_price = excluded.target_price,
                        stop_price = excluded.stop_price,
                        risk_reward = excluded.risk_reward,
                        cone_json = excluded.cone_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        p["symbol"].upper(),
                        p["as_of_date"],
                        float(p["pred_alpha_20d"]),
                        float(p["prob_outperform"]),
                        float(p.get("prob_tb_win", 50.0)),
                        float(p["expected_return_20d"]),
                        float(p["vol_20d_ann"]),
                        p.get("target_price"),
                        p.get("stop_price"),
                        p.get("risk_reward"),
                        json.dumps(p["cone"]),
                        p["updated_at"],
                    ),
                )
            if oos_rows is not None:
                conn.execute("DELETE FROM ml_oos_history")
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO ml_oos_history (
                        date, symbol, pred_alpha, prob_tb_win,
                        actual_fwd_20d, bench_fwd_20d, tb_outcome
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            r["date"],
                            r["symbol"],
                            float(r["pred_alpha"]),
                            float(r["prob_tb_win"]),
                            r.get("actual_fwd_20d"),
                            r.get("bench_fwd_20d"),
                            r.get("tb_outcome"),
                        )
                        for r in oos_rows
                    ],
                )

    def get_latest_ml_run(self) -> dict[str, Any] | None:
        with self.connect() as conn:
            r = conn.execute("SELECT * FROM ml_runs ORDER BY id DESC LIMIT 1").fetchone()
        if not r:
            return None
        d = dict(r)
        d["feature_importances"] = json.loads(d.pop("feature_importances_json") or "[]")
        d["folds"] = json.loads(d.pop("folds_json") or "[]")
        d["regime_stats"] = json.loads(d.pop("regime_stats_json") or "{}")
        return d

    def get_ml_predictions(self) -> dict[str, dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM ml_predictions").fetchall()
        out = {}
        for r in rows:
            d = dict(r)
            d["cone"] = json.loads(d.pop("cone_json") or "[]")
            out[d["symbol"]] = d
        return out

    def get_ml_prediction(self, symbol: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            r = conn.execute("SELECT * FROM ml_predictions WHERE symbol = ?", (symbol.upper(),)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["cone"] = json.loads(d.pop("cone_json") or "[]")
        return d

    def get_ohlcv_tail(self, symbol: str, limit: int = 2) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT date, open, high, low, close, volume FROM ohlcv WHERE symbol = ? ORDER BY date DESC LIMIT ?",
                (symbol.upper(), int(limit)),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_oos_history(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM ml_oos_history ORDER BY date ASC, symbol ASC").fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Prediction Journal & Alerts
    # ------------------------------------------------------------------
    def upsert_journal_entries(self, entries: list[dict[str, Any]]) -> int:
        if not entries:
            return 0
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO prediction_journal (
                    id, logged_date, symbol, entry_price, composite_score,
                    ml_tb_prob, ml_alpha_20d, target_price, stop_price,
                    conviction, regime, is_live, ret_5d, ret_20d, ret_60d,
                    spy_ret_20d, outcome, graded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    entry_price = excluded.entry_price,
                    composite_score = excluded.composite_score,
                    ml_tb_prob = excluded.ml_tb_prob,
                    ml_alpha_20d = excluded.ml_alpha_20d,
                    target_price = excluded.target_price,
                    stop_price = excluded.stop_price,
                    conviction = excluded.conviction,
                    regime = excluded.regime,
                    ret_5d = excluded.ret_5d,
                    ret_20d = excluded.ret_20d,
                    ret_60d = excluded.ret_60d,
                    spy_ret_20d = excluded.spy_ret_20d,
                    outcome = excluded.outcome,
                    graded_at = excluded.graded_at
                """,
                [
                    (
                        e["id"],
                        e["logged_date"],
                        e["symbol"],
                        float(e["entry_price"]),
                        float(e["composite_score"]),
                        float(e["ml_tb_prob"]),
                        float(e["ml_alpha_20d"]),
                        e.get("target_price"),
                        e.get("stop_price"),
                        e["conviction"],
                        e.get("regime", "Normal"),
                        1 if e.get("is_live", True) else 0,
                        e.get("ret_5d"),
                        e.get("ret_20d"),
                        e.get("ret_60d"),
                        e.get("spy_ret_20d"),
                        e.get("outcome", "Open"),
                        e.get("graded_at"),
                    )
                    for e in entries
                ],
            )
        return len(entries)

    def get_journal_entries(self, limit: int = 200) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM prediction_journal ORDER BY logged_date DESC, composite_score DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["is_live"] = bool(d["is_live"])
            out.append(d)
        return out

    def replace_alerts(self, items: list[dict[str, Any]]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM alerts")
            if not items:
                return
            conn.executemany(
                """
                INSERT OR REPLACE INTO alerts (
                    id, created_at, symbol, alert_type, severity, title, detail, acknowledged
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        it["id"],
                        it["created_at"],
                        it["symbol"],
                        it["alert_type"],
                        it["severity"],
                        it["title"],
                        it["detail"],
                        1 if it.get("acknowledged") else 0,
                    )
                    for it in items
                ],
            )

    def get_alerts(self, limit: int = 80) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["acknowledged"] = bool(d["acknowledged"])
            out.append(d)
        return out

    def acknowledge_alert(self, alert_id: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Paper-Trading / Forward-Proof Ledger
    # ------------------------------------------------------------------
    def get_paper_account(self) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM paper_account WHERE id = 1").fetchone()
        return dict(row) if row else None

    def get_paper_positions(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM paper_positions ORDER BY symbol").fetchall()
        return [dict(r) for r in rows]

    def get_paper_trades(self, limit: int = 120) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM paper_trades ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def save_paper_rebalance(
        self,
        account: dict[str, Any],
        positions: list[dict[str, Any]],
        new_trades: list[dict[str, Any]],
        lots: list[dict[str, Any]] | None = None,
        sleeves: list[dict[str, Any]] | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO paper_account (
                    id, started_at, last_rebalance_at, initial_capital, cash_inr,
                    realized_pnl_inr, total_fees_inr, est_stcg_tax_inr,
                    bench_entry_price, risk_profile, top_k,
                    peak_nav_inr, dd_brake_active, liquidcase_inr, goldbees_inr,
                    bench_tri_entry_price, est_ltcg_tax_inr, stcl_carry_inr, ltcl_carry_inr
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    account["started_at"],
                    account["last_rebalance_at"],
                    float(account["initial_capital"]),
                    float(account["cash_inr"]),
                    float(account["realized_pnl_inr"]),
                    float(account["total_fees_inr"]),
                    float(account["est_stcg_tax_inr"]),
                    float(account["bench_entry_price"]),
                    str(account["risk_profile"]),
                    int(account["top_k"]),
                    float(account.get("peak_nav_inr", account["initial_capital"])),
                    1 if account.get("dd_brake_active") else 0,
                    float(account.get("liquidcase_inr", account["cash_inr"])),
                    float(account.get("goldbees_inr", 0.0)),
                    account.get("bench_tri_entry_price"),
                    float(account.get("est_ltcg_tax_inr", 0.0)),
                    float(account.get("stcl_carry_inr", 0.0)),
                    float(account.get("ltcl_carry_inr", 0.0)),
                ),
            )
            conn.execute("DELETE FROM paper_positions")
            if positions:
                conn.executemany(
                    """
                    INSERT INTO paper_positions (
                        symbol, sector, shares, avg_cost, entry_date, target_price, stop_price
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            p["symbol"],
                            p.get("sector", "Unknown"),
                            int(p["shares"]),
                            float(p["avg_cost"]),
                            str(p["entry_date"]),
                            p.get("target_price"),
                            p.get("stop_price"),
                        )
                        for p in positions
                    ],
                )
            if new_trades:
                conn.executemany(
                    """
                    INSERT INTO paper_trades (
                        executed_at, trade_date, symbol, action, shares, price,
                        gross_value, fee_inr, realized_pnl_inr, reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            t["executed_at"],
                            t["trade_date"],
                            t["symbol"],
                            t["action"],
                            int(t["shares"]),
                            float(t["price"]),
                            float(t["gross_value"]),
                            float(t.get("fee_inr", 0.0)),
                            float(t.get("realized_pnl_inr", 0.0)),
                            str(t.get("reason", "")),
                        )
                        for t in new_trades
                    ],
                )
            if lots is not None:
                conn.execute("DELETE FROM paper_lots")
                if lots:
                    conn.executemany(
                        """
                        INSERT INTO paper_lots (
                            symbol, buy_date, shares_remaining, unit_cost_inr, sleeve_id, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                l["symbol"],
                                l["buy_date"],
                                float(l["shares_remaining"]),
                                float(l["unit_cost_inr"]),
                                int(l.get("sleeve_id", 1)),
                                str(l.get("created_at", account["last_rebalance_at"])),
                            )
                            for l in lots
                        ],
                    )
            if sleeves is not None:
                for sl in sleeves:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO paper_sleeves (
                            sleeve_id, last_rebalance_date, next_rebalance_date, status, target_weight_pct
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            int(sl["sleeve_id"]),
                            sl.get("last_rebalance_date"),
                            sl.get("next_rebalance_date"),
                            str(sl.get("status", "scheduled")),
                            float(sl.get("target_weight_pct", 25.0)),
                        ),
                    )

    def get_paper_lots(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM paper_lots ORDER BY buy_date ASC, id ASC").fetchall()
        return [dict(r) for r in rows]

    def get_paper_sleeves(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM paper_sleeves ORDER BY sleeve_id ASC").fetchall()
        return [dict(r) for r in rows]

    def record_paper_nav(self, snap: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO paper_nav_history (
                    date, nav_inr, equity_inr, liquidcase_inr, goldbees_inr,
                    peak_nav_inr, dd_pct, dd_brake_active, bench_tri_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snap["date"],
                    float(snap["nav_inr"]),
                    float(snap["equity_inr"]),
                    float(snap["liquidcase_inr"]),
                    float(snap["goldbees_inr"]),
                    float(snap["peak_nav_inr"]),
                    float(snap.get("dd_pct", 0.0)),
                    1 if snap.get("dd_brake_active") else 0,
                    snap.get("bench_tri_price"),
                ),
            )

    def get_paper_nav_history(self, limit: int = 120) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM paper_nav_history ORDER BY date ASC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_ohlcv_latest_dates(self) -> dict[str, str]:
        with self.connect() as conn:
            rows = conn.execute("SELECT symbol, MAX(date) AS max_date FROM ohlcv GROUP BY symbol").fetchall()
        return {str(r["symbol"]): str(r["max_date"])[:10] for r in rows if r["max_date"]}

    def get_macro_latest_dates(self) -> dict[str, str]:
        with self.connect() as conn:
            rows = conn.execute("SELECT symbol, MAX(date) AS max_date FROM macro_series GROUP BY symbol").fetchall()
        return {str(r["symbol"]): str(r["max_date"])[:10] for r in rows if r["max_date"]}

    def reset_paper_account(self) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM paper_account")
            conn.execute("DELETE FROM paper_positions")
            conn.execute("DELETE FROM paper_trades")
            conn.execute("DELETE FROM paper_lots")
            conn.execute("DELETE FROM paper_sleeves")
            conn.execute("DELETE FROM paper_nav_history")

    # ------------------------------------------------------------------
    # Official NSE Institutional Money Signals (Delivery %, FII OI, Bulk/Block)
    # ------------------------------------------------------------------
    def upsert_nse_delivery(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO nse_delivery (
                    symbol, date, close_price, traded_qty, deliv_qty,
                    deliv_pct, turnover_lacs, no_of_trades, avg_trade_val_inr
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r["symbol"],
                        r["date"],
                        float(r.get("close_price", 0.0)),
                        int(r.get("traded_qty", 0)),
                        int(r.get("deliv_qty", 0)),
                        float(r.get("deliv_pct", 0.0)),
                        float(r.get("turnover_lacs", 0.0)),
                        int(r.get("no_of_trades", 0)),
                        float(r.get("avg_trade_val_inr", 0.0)),
                    )
                    for r in rows
                ],
            )

    def get_nse_delivery_recent(self, days: int = 20) -> list[dict[str, Any]]:
        with self.connect() as conn:
            dates = [
                r["date"]
                for r in conn.execute(
                    "SELECT DISTINCT date FROM nse_delivery ORDER BY date DESC LIMIT ?",
                    (days,),
                ).fetchall()
            ]
            if not dates:
                return []
            placeholders = ",".join("?" for _ in dates)
            rows = conn.execute(
                f"SELECT * FROM nse_delivery WHERE date IN ({placeholders}) ORDER BY date ASC",
                dates,
            ).fetchall()
            return [dict(r) for r in rows]

    def upsert_nse_fii_oi(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO nse_fii_oi (
                    date, fii_idx_fut_long, fii_idx_fut_short, fii_idx_long_pct,
                    fii_net_idx_fut, dii_net_idx_fut, client_net_idx_fut, fii_stk_fut_net
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r["date"],
                        int(r.get("fii_idx_fut_long", 0)),
                        int(r.get("fii_idx_fut_short", 0)),
                        float(r.get("fii_idx_long_pct", 50.0)),
                        int(r.get("fii_net_idx_fut", 0)),
                        int(r.get("dii_net_idx_fut", 0)),
                        int(r.get("client_net_idx_fut", 0)),
                        int(r.get("fii_stk_fut_net", 0)),
                    )
                    for r in rows
                ],
            )

    def get_nse_fii_oi(self, limit: int = 30) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM nse_fii_oi ORDER BY date DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def replace_nse_bulk_deals(self, deals: list[dict[str, Any]]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM nse_bulk_deals")
            if not deals:
                return
            conn.executemany(
                """
                INSERT OR REPLACE INTO nse_bulk_deals (
                    id, date, deal_type, symbol, client_name, action, quantity, price, value_cr
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        d["id"],
                        d["date"],
                        d["deal_type"],
                        d["symbol"],
                        d["client_name"],
                        d["action"],
                        int(d.get("quantity", 0)),
                        float(d.get("price", 0.0)),
                        float(d.get("value_cr", 0.0)),
                    )
                    for d in deals
                ],
            )

    def get_nse_bulk_deals(self, limit: int = 60) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM nse_bulk_deals ORDER BY value_cr DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_nse_delivery_history(self) -> dict[str, dict[str, float]]:
        """Return {symbol: {date: deliv_pct}} across all stored NSE delivery rows."""
        out: dict[str, dict[str, float]] = {}
        with self.connect() as conn:
            rows = conn.execute("SELECT symbol, date, deliv_pct FROM nse_delivery ORDER BY date ASC").fetchall()
        for r in rows:
            sym = str(r["symbol"]).upper()
            out.setdefault(sym, {})[str(r["date"])[:10]] = float(r["deliv_pct"])
        return out

    # ------------------------------------------------------------------
    # Phase 2: Strictly Immutable Point-In-Time (PIT) Fundamentals Store
    # ------------------------------------------------------------------
    def insert_pit_fundamentals_snapshots(self, rows: list[dict[str, Any]]) -> int:
        """
        Insert immutable PIT fundamental snapshots using INSERT OR IGNORE on (symbol, as_of_date).
        Existing historical snapshots are NEVER overwritten (zero restatement lookahead).
        """
        if not rows:
            return 0
        with self.connect() as conn:
            before = conn.total_changes
            conn.executemany(
                """
                INSERT OR IGNORE INTO pit_fundamentals_history (
                    symbol, as_of_date, filing_date, trailing_pe, ev_to_ebitda,
                    fcf_yield, gross_profit_to_assets, accruals, roe,
                    debt_to_equity, market_cap_cr, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(r["symbol"]).upper(),
                        str(r["as_of_date"])[:10],
                        str(r["filing_date"])[:10] if r.get("filing_date") else None,
                        r.get("trailing_pe"),
                        r.get("ev_to_ebitda"),
                        r.get("fcf_yield"),
                        r.get("gross_profit_to_assets"),
                        r.get("accruals"),
                        r.get("roe"),
                        r.get("debt_to_equity"),
                        r.get("market_cap_cr"),
                        str(r.get("source", "snapshot")),
                        str(r["created_at"]),
                    )
                    for r in rows
                ],
            )
            return int(conn.total_changes - before)

    def get_pit_fundamentals_as_of(self, as_of_date: str) -> dict[str, dict[str, Any]]:
        """Return latest strictly causal PIT fundamental snapshot per symbol with as_of_date <= query date."""
        cutoff = str(as_of_date)[:10]
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT p.*
                FROM pit_fundamentals_history p
                INNER JOIN (
                    SELECT symbol, MAX(as_of_date) AS max_dt
                    FROM pit_fundamentals_history
                    WHERE as_of_date <= ?
                    GROUP BY symbol
                ) latest
                ON p.symbol = latest.symbol AND p.as_of_date = latest.max_dt
                """,
                (cutoff,),
            ).fetchall()
        return {str(r["symbol"]).upper(): dict(r) for r in rows}

    def get_pit_fundamentals_count(self) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM pit_fundamentals_history").fetchone()
        return int(row["cnt"]) if row else 0
