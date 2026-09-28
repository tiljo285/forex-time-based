import io
import os
import sqlite3
import tempfile
import zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Forex Time-Candle Breakout Backtester",
    page_icon="💱",
    layout="wide"
)

MAX_FLAT_PCT = 5.0

# ---- Bundled DB filenames (forex version) ----
DB_ZIP  = "forex_data.zip"
DB_FILE = "forex_data.db"

# ============================================================
# AUTO-EXTRACT DB FROM ZIP (for Streamlit Cloud / mobile)
# ============================================================
@st.cache_resource(show_spinner=False)
def ensure_db_extracted():
    """Extract DB from ZIP if not already extracted. Returns DB path or None."""
    if os.path.exists(DB_FILE):
        return DB_FILE
    if os.path.exists(DB_ZIP):
        try:
            with zipfile.ZipFile(DB_ZIP, "r") as z:
                z.extractall(".")
            return DB_FILE if os.path.exists(DB_FILE) else None
        except Exception:
            return None
    return None


# ============================================================
# SYMBOL DISCOVERY
# ============================================================
@st.cache_data(show_spinner=False)
def list_available_symbols(db_source, resolution):
    try:
        if isinstance(db_source, io.BytesIO):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
                tmp.write(db_source.read())
                tmp_path = tmp.name
            conn = sqlite3.connect(tmp_path)
        else:
            conn = sqlite3.connect(db_source)

        df = pd.read_sql_query(
            "SELECT DISTINCT symbol FROM candles WHERE resolution = ? ORDER BY symbol",
            conn, params=(resolution,)
        )
        conn.close()
        return df["symbol"].tolist()
    except Exception:
        return []


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(show_spinner=False)
def load_candles(db_source, symbol, resolution):
    if isinstance(db_source, io.BytesIO):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            tmp.write(db_source.read())
            tmp_path = tmp.name
        conn = sqlite3.connect(tmp_path)
    else:
        conn = sqlite3.connect(db_source)

    query = """
        SELECT time, open, high, low, close, volume
        FROM candles
        WHERE symbol = ? AND resolution = ?
        ORDER BY time ASC
    """
    df = pd.read_sql_query(query, conn, params=(symbol, resolution))
    conn.close()

    if df.empty:
        return df

    if df["time"].iloc[0] > 1e11:
        df["dt_utc"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    else:
        df["dt_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)

    df["dt_ist"] = df["dt_utc"].dt.tz_convert("Asia/Kolkata")
    df = df.set_index("dt_ist").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def resample_ohlc(df, tf_minutes):
    df_utc = df.copy()
    df_utc.index = df_utc.index.tz_convert("UTC")
    res = df_utc.resample(f"{tf_minutes}min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()
    res.index = res.index.tz_convert("Asia/Kolkata")
    return res


def generate_anchors(tf_minutes):
    anchors = []
    start = 5 * 60 + 30
    for offset in range(0, 24 * 60, tf_minutes):
        t = (start + offset) % (24 * 60)
        anchors.append((t // 60, t % 60))
    return anchors


def data_quality_report(df, freq="M"):
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    work["flat"] = work["high"] == work["low"]
    work["period"] = work.index.to_period(freq).astype(str)

    grouped = work.groupby("period")["flat"].agg(["sum", "count"])
    grouped["Flat %"] = (grouped["sum"] / grouped["count"] * 100).round(2)
    grouped["Clean %"] = (100 - grouped["Flat %"]).round(2)

    return (grouped[["sum", "count", "Flat %", "Clean %"]]
            .rename(columns={"sum": "Flat Bars", "count": "Total Bars"})
            .rename_axis("Month")
            .reset_index())


def find_last_clean_month(df, max_flat_pct=MAX_FLAT_PCT):
    if df.empty:
        return None
    dq = data_quality_report(df, freq="M")
    clean = dq.loc[dq["Flat %"] <= max_flat_pct, "Month"].tolist()
    if not clean:
        return None
    return pd.Period(clean[-1], freq="M").end_time.date()


# ============================================================
# BACKTEST — COARSE
# ============================================================
def backtest_anchor(df_tf, tf_minutes, anchor_hour, anchor_min,
                    risk_pct, account, fee_pct, slippage,
                    target_r=2.0, hold_hours=24):
    mask = (df_tf.index.hour == anchor_hour) & (df_tf.index.minute == anchor_min)
    anchors = df_tf[mask]
    if anchors.empty:
        return pd.DataFrame()

    times_arr = df_tf.index
    highs_arr = df_tf["high"].values
    lows_arr  = df_tf["low"].values
    opens_arr = df_tf["open"].values
    closes_arr = df_tf["close"].values

    trades = []
    for ts, candle in anchors.iterrows():
        h, l = candle["high"], candle["low"]
        if h - l <= 0:
            continue

        start_time = ts + pd.Timedelta(minutes=tf_minutes)
        end_time   = ts + pd.Timedelta(hours=hold_hours)
        s = times_arr.searchsorted(start_time, side="left")
        e = times_arr.searchsorted(end_time, side="right")
        if s >= e:
            continue

        fh = highs_arr[s:e]
        fl = lows_arr[s:e]
        fo = opens_arr[s:e]
        fc = closes_arr[s:e]

        lb = fh >= h
        sb = fl <= l
        li = int(np.argmax(lb)) if lb.any() else -1
        si = int(np.argmax(sb)) if sb.any() else -1

        if li == -1 and si == -1:
            continue

        if li == -1:
            direction, ei = "short", si
        elif si == -1:
            direction, ei = "long", li
        elif li < si:
            direction, ei = "long", li
        elif si < li:
            direction, ei = "short", si
        else:
            bo = fo[li]
            direction = "long" if abs(bo - h) <= abs(bo - l) else "short"
            ei = li if direction == "long" else si

        if direction == "long":
            entry = h + slippage
            sl = l - slippage
            tp = entry + target_r * (entry - sl)
        else:
            entry = l - slippage
            sl = h + slippage
            tp = entry - target_r * (sl - entry)

        rpu = abs(entry - sl)
        if rpu <= 0:
            continue
        qty = (account * risk_pct) / rpu

        ph = fh[ei + 1:]
        pl = fl[ei + 1:]
        pc = fc[ei + 1:]

        if direction == "long":
            sl_hit = pl <= sl
            tp_hit = ph >= tp
        else:
            sl_hit = ph >= sl
            tp_hit = pl <= tp

        si2 = int(np.argmax(sl_hit)) if sl_hit.any() else -1
        ti2 = int(np.argmax(tp_hit)) if tp_hit.any() else -1

        if si2 == -1 and ti2 == -1:
            exit_p, reason = (pc[-1] if len(pc) else entry), "Time"
        elif si2 == -1:
            exit_p, reason = tp, "TP"
        elif ti2 == -1:
            exit_p, reason = sl, "SL"
        elif si2 < ti2:
            exit_p, reason = sl, "SL"
        elif ti2 < si2:
            exit_p, reason = tp, "TP"
        else:
            exit_p, reason = sl, "SL"

        if direction == "long":
            gross = (exit_p - entry) * qty
            r = (exit_p - entry) / rpu
        else:
            gross = (entry - exit_p) * qty
            r = (entry - exit_p) / rpu

        fees = (entry * qty + exit_p * qty) * fee_pct
        net = gross - fees

        trades.append({
            "date": ts.date(),
            "month": ts.strftime("%Y-%m"),
            "anchor": f"{anchor_hour:02d}:{anchor_min:02d}",
            "direction": direction,
            "entry": entry, "sl": sl, "tp": tp, "exit": exit_p,
            "exit_reason": reason,
            "r_mult": r,
            "pnl": net,
            "range_pts": h - l,
        })

    return pd.DataFrame(trades)


# ============================================================
# BACKTEST — FINE (with candle size filter)
# ============================================================
def backtest_anchor_fine(df_base, df_tf, tf_minutes, anchor_hour, anchor_min,
                         risk_pct, account, fee_pct, slippage,
                         target_r=2.0, hold_hours=24, require_close=False,
                         size_filter=None):
    mask = (df_tf.index.hour == anchor_hour) & (df_tf.index.minute == anchor_min)
    anchors = df_tf[mask]
    if anchors.empty:
        return pd.DataFrame()

    anchor_ranges = (anchors["high"] - anchors["low"]).values
    if size_filter is not None and len(anchor_ranges) > 0:
        pmin, pmax = size_filter
        p_low  = np.percentile(anchor_ranges, pmin)
        p_high = np.percentile(anchor_ranges, pmax)
    else:
        p_low, p_high = -np.inf, np.inf

    base_times = df_base.index
    base_high  = df_base["high"].values
    base_low   = df_base["low"].values
    base_close = df_base["close"].values

    trades = []
    for anchor_ts, anchor_candle in anchors.iterrows():
        h = anchor_candle["high"]
        l = anchor_candle["low"]
        if h - l <= 0:
            continue

        this_range = h - l
        if this_range < p_low or this_range > p_high:
            continue

        entry_start = anchor_ts + pd.Timedelta(minutes=tf_minutes)
        entry_end   = entry_start + pd.Timedelta(hours=hold_hours)
        s = base_times.searchsorted(entry_start, side="left")
        e = base_times.searchsorted(entry_end,   side="right")
        if s >= e:
            continue

        win_h = base_high[s:e]
        win_l = base_low[s:e]
        win_c = base_close[s:e]

        if require_close:
            hit_high = win_c > h
            hit_low  = win_c < l
        else:
            hit_high = win_h >= h
            hit_low  = win_l <= l

        first_high = int(np.argmax(hit_high)) if hit_high.any() else -1
        first_low  = int(np.argmax(hit_low))  if hit_low.any()  else -1

        if first_high == -1 and first_low == -1:
            continue

        if first_high == -1:
            direction, entry_bar = "short", first_low
        elif first_low == -1:
            direction, entry_bar = "long", first_high
        elif first_high < first_low:
            direction, entry_bar = "long", first_high
        elif first_low < first_high:
            direction, entry_bar = "short", first_low
        else:
            bo = win_c[first_high]
            direction = "long" if abs(bo - h) <= abs(bo - l) else "short"
            entry_bar = first_high if direction == "long" else first_low

        if direction == "long":
            entry = h + slippage
            sl    = l - slippage
            tp    = entry + target_r * (entry - sl)
        else:
            entry = l - slippage
            sl    = h + slippage
            tp    = entry - target_r * (sl - entry)

        rpu = abs(entry - sl)
        if rpu <= 0:
            continue
        qty = (account * risk_pct) / rpu

        ph = win_h[entry_bar + 1:]
        pl = win_l[entry_bar + 1:]
        pc = win_c[entry_bar + 1:]

        if direction == "long":
            sl_hit = pl <= sl
            tp_hit = ph >= tp
        else:
            sl_hit = ph >= sl
            tp_hit = pl <= tp

        si2 = int(np.argmax(sl_hit)) if sl_hit.any() else -1
        ti2 = int(np.argmax(tp_hit)) if tp_hit.any() else -1

        if si2 == -1 and ti2 == -1:
            exit_p, reason = (pc[-1] if len(pc) else entry), "Time"
        elif si2 == -1:
            exit_p, reason = tp, "TP"
        elif ti2 == -1:
            exit_p, reason = sl, "SL"
        elif si2 < ti2:
            exit_p, reason = sl, "SL"
        elif ti2 < si2:
            exit_p, reason = tp, "TP"
        else:
            exit_p, reason = sl, "SL"

        if direction == "long":
            gross = (exit_p - entry) * qty
            r = (exit_p - entry) / rpu
        else:
            gross = (entry - exit_p) * qty
            r = (entry - exit_p) / rpu

        fees = (entry * qty + exit_p * qty) * fee_pct
        net = gross - fees

        trades.append({
            "date": anchor_ts.date(),
            "month": anchor_ts.strftime("%Y-%m"),
            "anchor": f"{anchor_hour:02d}:{anchor_min:02d}",
            "direction": direction,
            "entry": entry, "sl": sl, "tp": tp, "exit": exit_p,
            "exit_reason": reason,
            "r_mult": r,
            "pnl": net,
            "range_pts": this_range,
        })

    return pd.DataFrame(trades)


def compute_metrics(trades, account):
    if trades.empty:
        return None

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]

    gp = wins["pnl"].sum()
    gl = abs(losses["pnl"].sum())
    pf = gp / gl if gl > 0 else float("inf")

    eq = account + trades["pnl"].cumsum()
    rm = eq.cummax()
    dd = ((eq - rm) / rm).min() * 100

    mc, cur = 0, 0
    for p in trades["pnl"].values:
        if p <= 0:
            cur += 1
            mc = max(mc, cur)
        else:
            cur = 0

    monthly = trades.groupby("month")["pnl"].sum()
    pct_pos = (monthly > 0).mean() * 100 if len(monthly) else 0

    return {
        "Trades": len(trades),
        "Win %": len(wins) / len(trades) * 100,
        "Profit Factor": pf,
        "Net PnL ($)": trades["pnl"].sum(),
        "Expectancy (R)": trades["r_mult"].mean(),
        "Max DD %": dd,
        "Max Consec Losses": mc,
        "Months +ve %": pct_pos,
    }


# ============================================================
# CANDLE SIZE ANALYSIS
# ============================================================
def candle_size_analysis(trades, n_buckets=5):
    if trades.empty or "range_pts" not in trades.columns:
        return pd.DataFrame(), trades

    t = trades.copy()
    t["size_bucket"] = pd.qcut(
        t["range_pts"], q=n_buckets,
        labels=[f"Q{i+1}" for i in range(n_buckets)],
        duplicates="drop"
    )

    def bucket_metrics(g):
        wins = g[g["pnl"] > 0]
        losses = g[g["pnl"] <= 0]
        gp = wins["pnl"].sum()
        gl = abs(losses["pnl"].sum())
        pf = gp / gl if gl > 0 else float("inf")
        monthly = g.groupby("month")["pnl"].sum()
        months_pos = (monthly > 0).mean() * 100 if len(monthly) else 0
        return pd.Series({
            "Trades": len(g),
            "Avg Range": g["range_pts"].mean(),
            "Win %": (g["pnl"] > 0).mean() * 100,
            "Profit Factor": pf,
            "Expectancy (R)": g["r_mult"].mean(),
            "Months +ve %": months_pos,
        })

    summary = (t.groupby("size_bucket", observed=True)
               .apply(bucket_metrics)
               .reset_index())
    summary.columns = ["Size Bucket", "Trades", "Avg Range",
                       "Win %", "Profit Factor", "Expectancy (R)", "Months +ve %"]
    return summary, t


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Configuration")

# Try to auto-extract the bundled DB first
bundled_db_path = ensure_db_extracted()

if bundled_db_path:
    db_mode = st.sidebar.radio(
        "Database Source",
        ["Bundled DB (auto-extracted)", "Upload .db file"]
    )
    if db_mode == "Bundled DB (auto-extracted)":
        db_source = bundled_db_path
        st.sidebar.caption(f"✅ Using bundled `{DB_FILE}`")
    else:
        db_source = st.sidebar.file_uploader(
            "Upload SQLite DB", type=["db", "sqlite", "sqlite3"]
        )
else:
    st.sidebar.warning(
        f"⚠️ Bundled `{DB_ZIP}` not found. Upload a .db file to proceed."
    )
    db_source = st.sidebar.file_uploader(
        "Upload SQLite DB", type=["db", "sqlite", "sqlite3"]
    )

base_res = st.sidebar.selectbox(
    "Base Resolution (for resampling)",
    ["1m", "5m", "15m", "1h"],
    index=1
)
base_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}[base_res]

if not db_source:
    st.info("Load or upload your SQLite DB from the sidebar.")
    st.stop()

available_symbols = list_available_symbols(db_source, base_res)

if not available_symbols:
    st.sidebar.error(f"No symbols found in DB for resolution `{base_res}`.")
    st.stop()

symbol = st.sidebar.selectbox(
    "Symbol",
    options=available_symbols,
    index=available_symbols.index("EURUSD") if "EURUSD" in available_symbols else 0,
    help=f"{len(available_symbols)} symbols available in {base_res} data."
)

st.sidebar.caption(f"📊 {len(available_symbols)} symbols in DB: "
                   f"{', '.join(available_symbols[:5])}"
                   f"{' ...' if len(available_symbols) > 5 else ''}")

ALL_TFS = [15, 30, 60, 120, 240, 360]
available_tfs = [t for t in ALL_TFS if t >= base_minutes]
selected_tfs = st.sidebar.multiselect(
    "Test Timeframes (min)",
    available_tfs,
    default=[30, 60, 120, 240] if base_minutes <= 30 else [60, 120, 240],
    help="Forex often rewards 30m–120m more than crypto does. Test a few."
)

selected_rrs = st.sidebar.multiselect(
    "Target R:R ratios to test",
    options=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
    default=[1.0, 2.0, 3.0, 4.0, 5.0],
    help="Every (TF × Anchor × R:R) combination will be tested and ranked. "
         "More R:R values = longer runtime but wider search."
)

hold_hours = st.sidebar.number_input(
    "Hold Window (hours from anchor)",
    value=24, min_value=1, max_value=72
)

st.sidebar.markdown("---")
st.sidebar.subheader("Entry Detection")
entry_mode = st.sidebar.radio(
    "Mode",
    ["Fine (base bars — realistic)", "Coarse (TF bars — fast)"],
    index=0,
    help="Fine walks forward on base bars (matches live entry). "
         "Coarse walks on resampled TF bars (faster)."
)
require_close = st.sidebar.checkbox(
    "Require base-bar CLOSE beyond level",
    value=True,
    help="Stricter: the base bar must CLOSE beyond the anchor H/L, not just wick."
)

# ---- Candle size filter ----
st.sidebar.markdown("---")
st.sidebar.subheader("📏 Candle Size Filter")
size_filter_on = st.sidebar.checkbox(
    "Filter anchors by candle range",
    value=False,
    help="Only trade anchors whose H-L range falls within the selected "
         "percentile band. Computed per (TF, anchor)."
)
if size_filter_on:
    size_min_pct = st.sidebar.slider(
        "Min percentile (0 = smallest)", 0, 100, 0, 5,
    )
    size_max_pct = st.sidebar.slider(
        "Max percentile (100 = largest)", 0, 100, 100, 5,
    )
    if size_min_pct >= size_max_pct:
        st.sidebar.error("Min must be < Max")
        size_filter = None
    else:
        size_filter = (size_min_pct, size_max_pct)
    st.sidebar.caption(
        f"Only anchors between {size_min_pct}th and {size_max_pct}th "
        f"percentile of range size are traded."
    )
else:
    size_filter = None

st.sidebar.markdown("---")
st.sidebar.subheader("Risk & Costs")
risk_pct = st.sidebar.number_input("Risk per Trade (%)", value=1.0, step=0.1) / 100
account  = st.sidebar.number_input("Starting Capital ($)", value=10000, step=1000)
fee_pct  = st.sidebar.number_input(
    "Round-trip cost (%)",
    value=0.005, step=0.001, format="%.4f",
    help="0.005% ≈ 0.5 pip on EURUSD at 1.08. Retail ECN typically 0.002–0.01%."
) / 100
slippage = st.sidebar.number_input(
    "Slippage per side (price units)",
    value=0.00005, step=0.00001, format="%.5f",
    help="EURUSD/GBPUSD/AUDUSD: 0.00005 = 0.5 pip. "
         "USDJPY: 0.005 = 0.5 pip. XAUUSD: 0.05 = 5 cents."
)

# ============================================================
# MAIN
# ============================================================
st.title("💱 Forex Time-Candle Breakout Backtester")
st.caption(
    "Mark a time-candle's H/L. Enter on either-side breakout. "
    "SL = opposite extreme. Tests every (TF × Anchor × R:R) combination."
)

try:
    with st.spinner(f"Loading {symbol} candles..."):
        df_all = load_candles(db_source, symbol, base_res)
except Exception as e:
    st.error(f"Failed to load DB: {e}")
    st.stop()

if df_all.empty:
    st.warning(f"No data for {symbol} @ {base_res}.")
    st.stop()

min_d = df_all.index.min().date()
max_d = df_all.index.max().date()

last_clean = find_last_clean_month(df_all, MAX_FLAT_PCT)
default_end = min(max_d, last_clean) if last_clean else max_d
default_start = min_d

st.sidebar.markdown("---")
if last_clean:
    st.sidebar.caption(f"🟢 Auto-detected last clean month: **{last_clean}**")
else:
    st.sidebar.caption("⚠️ No clean month detected — check data quality.")

date_range = st.sidebar.date_input(
    "Date Range",
    value=(default_start, default_end),
    min_value=min_d,
    max_value=max_d,
)

if len(date_range) == 2:
    s_dt, e_dt = date_range
else:
    s_dt = e_dt = date_range[0]

start_ts = pd.Timestamp(s_dt, tz="Asia/Kolkata")
end_ts   = pd.Timestamp(e_dt, tz="Asia/Kolkata") + pd.Timedelta(days=1)
df_base  = df_all[(df_all.index >= start_ts) & (df_all.index < end_ts)]

if df_base.empty:
    st.warning("No data in range.")
    st.stop()

st.write(
    f"**Loaded** `{symbol}` @ {base_res}: **{len(df_base):,}** bars | "
    f"Range: **{s_dt} → {e_dt}**"
)

# ------------------------------------------------------------
# DATA QUALITY GUARD
# ------------------------------------------------------------
flat_mask = df_base["high"] == df_base["low"]
flat_pct  = flat_mask.mean() * 100

dq_col1, dq_col2, dq_col3 = st.columns(3)
dq_col1.metric("Total Bars", f"{len(df_base):,}")
dq_col2.metric("Flat Bars (O=H=L=C)", f"{flat_mask.sum():,}")
dq_col3.metric(
    "Flat %",
    f"{flat_pct:.2f}%",
    delta=f"limit {MAX_FLAT_PCT:.0f}%",
    delta_color="inverse"
)

if flat_pct > MAX_FLAT_PCT:
    st.error(
        f"🚫 **Data quality check FAILED** — {flat_pct:.1f}% of bars in the "
        f"selected range are flat (O=H=L=C). Corrupt feed — backtest will "
        f"be meaningless. Narrow the Date Range or re-download."
    )

    with st.expander("📅 Flat-bar % by month (find the cut-off)", expanded=True):
        dq = data_quality_report(df_base, freq="M")
        st.dataframe(
            dq.style
              .format({"Flat Bars": "{:,}", "Total Bars": "{:,}",
                       "Flat %": "{:.2f}", "Clean %": "{:.2f}"})
              .background_gradient(subset=["Flat %"], cmap="Reds"),
            use_container_width=True,
            hide_index=True,
        )
        clean_months = dq.loc[dq["Flat %"] <= MAX_FLAT_PCT, "Month"].tolist()
        if clean_months:
            st.success(
                f"✅ Last clean month in this range: **{clean_months[-1]}**. "
                f"Set your Date Range end date to the last day of that month."
            )
        else:
            st.warning("No clean months in the selected range.")

    st.stop()

if flat_pct > 1.0:
    st.warning(
        f"⚠️ {flat_pct:.2f}% of bars are flat. Under the {MAX_FLAT_PCT:.0f}% limit."
    )

st.markdown("---")

# ------------------------------------------------------------
# RUN MATRIX
# ------------------------------------------------------------
if st.button("🚀 Run Full Matrix Backtest", type="primary"):
    if not selected_tfs:
        st.warning("Select at least one timeframe.")
        st.stop()
    if not selected_rrs:
        st.warning("Select at least one R:R value.")
        st.stop()

    tfs_data = {}
    progress = st.progress(0.0, text="Resampling...")
    for i, tf in enumerate(selected_tfs):
        tfs_data[tf] = resample_ohlc(df_base, tf)
        progress.progress(
            (i + 1) / (len(selected_tfs) + 1),
            text=f"Resampled {tf}m ({len(tfs_data[tf]):,} bars)"
        )

    combos = []
    for tf in selected_tfs:
        for (ah, am) in generate_anchors(tf):
            for rr in selected_rrs:
                combos.append((tf, ah, am, rr))

    all_results = []
    all_trades = {}
    text = st.empty()

    use_fine = entry_mode.startswith("Fine")
    mode_label = "Fine" if use_fine else "Coarse"
    close_label = " (close-only)" if (use_fine and require_close) else ""
    size_label = f" | size {size_filter[0]}-{size_filter[1]}%" if size_filter else ""

    for i, (tf, ah, am, rr) in enumerate(combos):
        text.info(
            f"[{mode_label}{close_label}{size_label}] {symbol} — "
            f"TF={tf}m @ {ah:02d}:{am:02d} IST | RR 1:{rr:g}  "
            f"({i+1}/{len(combos)})"
        )
        if use_fine:
            trades = backtest_anchor_fine(
                df_base, tfs_data[tf], tf, ah, am,
                risk_pct, account, fee_pct, slippage,
                target_r=rr, hold_hours=hold_hours,
                require_close=require_close,
                size_filter=size_filter,
            )
        else:
            trades = backtest_anchor(
                tfs_data[tf], tf, ah, am,
                risk_pct, account, fee_pct, slippage,
                target_r=rr, hold_hours=hold_hours
            )

        if not trades.empty:
            m = compute_metrics(trades, account)
            if m:
                m["TF"] = tf
                m["Anchor"] = f"{ah:02d}:{am:02d}"
                m["RR"] = rr
                m["RR_Label"] = f"1:{rr:g}"
                all_results.append(m)
                all_trades[(tf, f"{ah:02d}:{am:02d}", rr)] = trades

    text.empty()
    progress.empty()

    if not all_results:
        st.error("No trades generated.")
        st.stop()

    res_df = pd.DataFrame(all_results)[
        ["TF", "Anchor", "RR", "RR_Label", "Trades", "Win %", "Profit Factor",
         "Expectancy (R)", "Net PnL ($)", "Max DD %", "Max Consec Losses", "Months +ve %"]
    ]
    st.session_state["results"] = res_df
    st.session_state["trades"]  = all_trades
    st.session_state["symbol"]  = symbol

    st.success(
        f"✅ {symbol} — {len(combos)} combinations tested "
        f"({mode_label}{close_label}{size_label}) — {len(res_df)} produced trades."
    )

    res_sorted = res_df.sort_values(
        "Profit Factor", ascending=False
    ).reset_index(drop=True)

    best = res_sorted.iloc[0]
    st.markdown(
        f"### 🏆 Best Combo ({symbol}): **TF = {int(best['TF'])}m @ "
        f"{best['Anchor']} IST | RR 1:{best['RR']:g}**  "
        f"→ PF **{best['Profit Factor']:.2f}**,  "
        f"Win **{best['Win %']:.1f}%**,  "
        f"Expectancy **{best['Expectancy (R)']:.3f} R**"
    )

    display_cols = ["TF", "Anchor", "RR_Label", "Trades", "Win %",
                    "Profit Factor", "Expectancy (R)", "Net PnL ($)",
                    "Max DD %", "Max Consec Losses", "Months +ve %"]
    st.subheader("📊 Ranked Results (by Profit Factor)")
    st.dataframe(
        res_sorted[display_cols].style
            .format({
                "RR_Label": "{}",
                "Win %": "{:.1f}", "Profit Factor": "{:.2f}",
                "Expectancy (R)": "{:.3f}", "Net PnL ($)": "${:.2f}",
                "Max DD %": "{:.1f}", "Months +ve %": "{:.1f}",
            })
            .background_gradient(subset=["Profit Factor"], cmap="Greens")
            .background_gradient(subset=["Expectancy (R)"], cmap="RdYlGn"),
        use_container_width=True, height=520
    )

    st.subheader("📊 R:R Summary — Average across all (TF, Anchor) combos")
    rr_summary = (res_df.groupby("RR_Label")
                  .agg({
                      "Profit Factor": "mean",
                      "Expectancy (R)": "mean",
                      "Win %": "mean",
                      "Months +ve %": "mean",
                      "Trades": "mean",
                  })
                  .reset_index()
                  .sort_values("Profit Factor", ascending=False)
                  .round(3))
    st.dataframe(rr_summary, use_container_width=True, hide_index=True)

    st.subheader("🔥 Heatmap — Profit Factor (TF × Anchor)")
    heatmap_rr = st.selectbox(
        "Select R:R to visualize",
        options=sorted(res_df["RR"].unique()),
        format_func=lambda v: f"1:{v:g}",
        index=0,
    )
    filtered = res_df[res_df["RR"] == heatmap_rr]

    pivot_pf = filtered.pivot_table(
        index="TF", columns="Anchor", values="Profit Factor"
    )
    fig_pf = px.imshow(
        pivot_pf, aspect="auto", color_continuous_scale="RdYlGn",
        origin="lower", text_auto=".2f",
        labels=dict(x="Anchor Time (IST)", y="Timeframe (min)", color="PF")
    )
    fig_pf.update_layout(height=380,
                         title=f"Profit Factor — RR 1:{heatmap_rr:g}")
    st.plotly_chart(fig_pf, use_container_width=True)

    pivot_r = filtered.pivot_table(
        index="TF", columns="Anchor", values="Expectancy (R)"
    )
    fig_r = px.imshow(
        pivot_r, aspect="auto", color_continuous_scale="RdYlGn",
        origin="lower", text_auto=".3f",
        labels=dict(x="Anchor Time (IST)", y="Timeframe (min)", color="R")
    )
    fig_r.update_layout(height=380,
                        title=f"Expectancy (R) — RR 1:{heatmap_rr:g}")
    st.plotly_chart(fig_r, use_container_width=True)

# ------------------------------------------------------------
# DETAILED VIEW
# ------------------------------------------------------------
if "results" in st.session_state and "trades" in st.session_state:
    st.markdown("---")
    st.header("🔍 Detailed Combination Inspector")

    res_df = st.session_state["results"]
    all_trades = st.session_state["trades"]
    active_symbol = st.session_state.get("symbol", symbol)
    res_sorted = res_df.sort_values(
        "Profit Factor", ascending=False
    ).reset_index(drop=True)

    st.caption(f"Inspecting: **{active_symbol}**")

    options = [
        f"TF={int(r['TF'])}m @ {r['Anchor']} IST | RR 1:{r['RR']:g}   "
        f"(PF {r['Profit Factor']:.2f} | "
        f"Win {r['Win %']:.1f}% | "
        f"Expectancy {r['Expectancy (R)']:.3f} R)"
        for _, r in res_sorted.iterrows()
    ]
    choice = st.selectbox("Select a combination to inspect:", options)
    idx = options.index(choice)
    row = res_sorted.iloc[idx]
    tf, anchor, rr = int(row["TF"]), row["Anchor"], float(row["RR"])

    trades = all_trades.get((tf, anchor, rr))
    if trades is None or trades.empty:
        st.warning("No trades for this combo.")
        st.stop()

    trades = trades.copy()
    trades["Equity"]  = account + trades["pnl"].cumsum()
    trades["Cum PnL"] = trades["pnl"].cumsum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Trades", len(trades))
    c2.metric("Win Rate", f"{row['Win %']:.1f}%")
    c3.metric("Profit Factor", f"{row['Profit Factor']:.2f}")
    c4.metric("Net PnL", f"${row['Net PnL ($)']:.2f}")
    c5.metric("Max DD", f"{row['Max DD %']:.1f}%")

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=list(range(len(trades))), y=trades["Equity"],
        mode="lines", line=dict(color="#00CC96", width=2),
        fill="tozeroy", name="Equity"
    ))
    fig_eq.update_layout(
        title=f"Equity — {active_symbol} — TF={tf}m @ {anchor} IST | RR 1:{rr:g}",
        xaxis_title="Trade #", yaxis_title="Balance ($)", height=380
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # ---------------- CANDLE SIZE ANALYSIS ----------------
    st.markdown("---")
    st.subheader("📏 Candle Size Analysis — Does anchor size matter?")

    size_summary, trades_with_bucket = candle_size_analysis(trades, n_buckets=5)

    if size_summary.empty:
        st.info("No size data available for this combo.")
    else:
        rmin = trades["range_pts"].min()
        rmax = trades["range_pts"].max()
        rmed = trades["range_pts"].median()
        s1, s2, s3 = st.columns(3)
        s1.metric("Smallest anchor", f"{rmin:.5f}")
        s2.metric("Median anchor",   f"{rmed:.5f}")
        s3.metric("Largest anchor",  f"{rmax:.5f}")

        st.dataframe(
            size_summary.style
                .format({"Avg Range": "{:.5f}", "Win %": "{:.1f}",
                         "Profit Factor": "{:.2f}", "Expectancy (R)": "{:.3f}",
                         "Months +ve %": "{:.1f}"})
                .background_gradient(subset=["Profit Factor"], cmap="Greens")
                .background_gradient(subset=["Expectancy (R)"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True
        )

        colA, colB = st.columns(2)
        with colA:
            fig_pf_b = px.bar(
                size_summary, x="Size Bucket", y="Profit Factor",
                color="Profit Factor", color_continuous_scale="RdYlGn",
                text_auto=".2f", title="Profit Factor by anchor size bucket"
            )
            fig_pf_b.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig_pf_b, use_container_width=True)
        with colB:
            fig_wr_b = px.bar(
                size_summary, x="Size Bucket", y="Win %",
                color="Win %", color_continuous_scale="RdYlGn",
                text_auto=".1f", title="Win % by anchor size bucket"
            )
            fig_wr_b.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig_wr_b, use_container_width=True)

        fig_sc = px.scatter(
            trades, x="range_pts", y="r_mult",
            color=trades["r_mult"] > 0,
            labels={"range_pts": "Anchor candle range",
                    "r_mult": "Trade R-multiple",
                    "color": "Winner"},
            color_discrete_map={True: "#00CC96", False: "#EF553B"},
            opacity=0.5, title="Every trade: anchor size vs outcome",
        )
        fig_sc.update_layout(height=400)
        st.plotly_chart(fig_sc, use_container_width=True)

        best_bucket = size_summary.loc[size_summary["Profit Factor"].idxmax()]
        worst_bucket = size_summary.loc[size_summary["Profit Factor"].idxmin()]
        st.info(
            f"**Best size bucket:** `{best_bucket['Size Bucket']}` "
            f"(PF {best_bucket['Profit Factor']:.2f}, "
            f"Win {best_bucket['Win %']:.1f}%, "
            f"Avg range {best_bucket['Avg Range']:.5f})\n\n"
            f"**Worst size bucket:** `{worst_bucket['Size Bucket']}` "
            f"(PF {worst_bucket['Profit Factor']:.2f}, "
            f"Win {worst_bucket['Win %']:.1f}%, "
            f"Avg range {worst_bucket['Avg Range']:.5f})\n\n"
            f"If the best bucket is clearly ahead, enable the sidebar "
            f"**Candle Size Filter** and re-run the matrix with the "
            f"corresponding percentile band."
        )

    st.markdown("---")
    st.subheader("📈 Monthly PnL")
    monthly = trades.groupby("month")["pnl"].sum().reset_index()
    monthly["color"] = monthly["pnl"].apply(
        lambda v: "#00CC96" if v >= 0 else "#EF553B"
    )
    fig_m = go.Figure(go.Bar(
        x=monthly["month"], y=monthly["pnl"],
        marker_color=monthly["color"]
    ))
    fig_m.update_layout(
        title="Monthly PnL", xaxis_title="Month",
        yaxis_title="PnL ($)", height=340
    )
    st.plotly_chart(fig_m, use_container_width=True)

    ec1, ec2 = st.columns(2)
    with ec1:
        ec = trades["exit_reason"].value_counts().reset_index()
        ec.columns = ["Reason", "Count"]
        st.plotly_chart(
            px.pie(ec, values="Count", names="Reason",
                   hole=0.45, title="Exit Reasons"),
            use_container_width=True
        )
    with ec2:
        dc = trades["direction"].value_counts().reset_index()
        dc.columns = ["Direction", "Count"]
        st.plotly_chart(
            px.pie(dc, values="Count", names="Direction",
                   hole=0.45, title="Long vs Short"),
            use_container_width=True
        )

    st.subheader("Trade Log")
    st.dataframe(
        trades[["date", "direction", "entry", "sl", "tp", "exit",
                "exit_reason", "r_mult", "range_pts", "pnl", "Equity"]]
        .style.format({
            "entry": "{:.5f}", "sl": "{:.5f}",
            "tp": "{:.5f}", "exit": "{:.5f}",
            "r_mult": "{:.2f} R",
            "range_pts": "{:.5f}",
            "pnl": "${:.2f}", "Equity": "${:.2f}",
        }),
        use_container_width=True, height=420
    )

    csv = trades.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Trade Log (CSV)", csv,
        file_name=f"trades_{active_symbol}_{tf}m_{anchor.replace(':', '')}_rr{rr:g}.csv",
        mime="text/csv"
    )