# app_unified.py — Unified FinGPT-M Chat (Charts + Financial Docs + Text), FinGPT LoRA only

import os
import re
import json
import time
import hashlib
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import sys
sys.path.insert(0, r"/content/FinGPT-M/fingpt/stock_chart_trends_analysis")

import torch
import gradio as gr
import pandas as pd
import yfinance as yf
import finnhub
from dotenv import load_dotenv
from datetime import date, datetime, timedelta

import nltk
from nltk.corpus import stopwords

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig
from peft import PeftModel

from google.colab import userdata

from StockChart_Trend_Prediction import StockChartTrendPredictor, StockChartMetadataExtractor
# import finance_rag_semantic as frs

# ── ENV / paths
load_dotenv(override=True)

HF_TOKEN = userdata.get("HF_TOKEN")
FINNHUB_API_KEY = userdata.get("FINNHUB_API_KEY")
if not HF_TOKEN: raise RuntimeError("HF_TOKEN not set")
if not FINNHUB_API_KEY: raise RuntimeError("FINNHUB_API_KEY not set")

YOLO_WEIGHTS = r"/content/FinGPT-M/fingpt/stock_chart_trends_analysis/best.pt"
os.environ["YOLO_MODEL_PATH"] = YOLO_WEIGHTS
os.environ["ULTRALYTICS_VERBOSE"] = "False"

# ── NLTK
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
EN_STOP = set(stopwords.words("english"))
EN_STOP.update({"buy","sell","hold","call","put","usd","nse","bse","nyse","nasdaq","market","stock","shares"})

# ── Finnhub
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# ── Model load (FinGPT LoRA only)
LLAMA_BASE = "meta-llama/Llama-2-7b-chat-hf"
LORA_ADAPTER = "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"
CACHE_DIR = r"E:/FinGPT/llama_cache"
OFFLOAD_DIR = r"E:/FinGPT/offload"

if torch.cuda.is_available():
    dtype = torch.bfloat16
    device_map = "auto"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=False,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
else:
    dtype = torch.float32
    device_map = "cpu"
    bnb_config = None

print("• Loading base model…")
base_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_BASE,
    token=HF_TOKEN,
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
    device_map=device_map,
    torch_dtype=dtype,
    offload_folder=OFFLOAD_DIR,
    quantization_config=bnb_config if torch.cuda.is_available() else None,
)

print("• Applying LoRA adapter…")
model = PeftModel.from_pretrained(
    base_model,
    LORA_ADAPTER,
    offload_folder=OFFLOAD_DIR,
    cache_dir=CACHE_DIR,
).eval()

print("• Loading tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(LLAMA_BASE, token=HF_TOKEN, cache_dir=CACHE_DIR, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# ── Constants
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
OUTPUT_BEGIN, OUTPUT_END = "[OUTPUT_BEGIN]", "[OUTPUT_END]"
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DOC_EXT = {".pdf", ".docx", ".txt", ".md", ".csv", ".html", ".htm", ".pptx"}

# ── Helpers
def _dbg(msg: str):
    """
    Simple debug printer for FinGPT-M.
    Prints messages to console prefixed with [FinGPT-M].
    """
    print(f"[FinGPT-M] {msg}")
    
def _ensure_forecast_section(ans: str, final_output: Dict[str, Any]) -> str:
    """Guarantee a '### Forecast & Analysis' section exists; append a safe baseline if missing."""
    if "### Forecast & Analysis" in ans:
        return ans

    tkr = final_output.get("ticker") or final_output.get("company_ticker") or "—"
    pr = final_output.get("price_range") or [None, None]
    lo, hi = pr if isinstance(pr, (list, tuple)) and len(pr) == 2 else (None, None)

    baseline = (
        "\n\n### Forecast & Analysis\n"
        f"- Expect range-bound action around **{lo}–{hi}** in the near term unless **{tkr}** breaks recent extremes.\n"
        "- Watch for volume expansion and a decisive close beyond the recent high/low to confirm direction.\n"
    )
    return ans.strip() + baseline


def _between_markers(text: str) -> str:
    if OUTPUT_BEGIN in text: text = text.split(OUTPUT_BEGIN, 1)[1]
    if OUTPUT_END in text: text = text.split(OUTPUT_END, 1)[0]
    return text.strip()

def get_curday() -> str:
    return date.today().strftime("%Y-%m-%d")

def _yf_symbol(ticker: str, exchange: Optional[str]) -> str:
    if not ticker: return ticker
    t = ticker.upper().replace(" ","")
    ex = (exchange or "").upper()
    if ex in {"NSE", "NSEI", "INDIA"} and not t.endswith(".NS"): return f"{t}.NS"
    if ex == "BSE" and not t.endswith(".BO"): return f"{t}.BO"
    return t

# ---- NEW: symbol lookup helpers
def _finnhub_lookup_symbol(query: str) -> List[Dict[str, Any]]:
    try:
        r = finnhub_client.symbol_lookup(query)
        return (r or {}).get("result", []) or []
    except Exception as e:
        print(f"[symbol_lookup] {e}")
        return []

def _normalize_exchange_hint(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.upper().strip()
    if s in {"NSE", "NSEI", "INDIA"}: return "NSE"
    if s in {"BSE"}: return "BSE"
    if s in {"NYSE", "NASDAQ", "NASD"}: return s
    return s

def _pick_symbol_from_lookup(results: List[Dict[str, Any]], exchange_hint: Optional[str]) -> Optional[str]:
    if not results: return None
    exch = _normalize_exchange_hint(exchange_hint)
    if exch in {"NSE", "BSE"}:
        suffix = ".NS" if exch == "NSE" else ".BO"
        for r in results:
            sym = (r.get("symbol") or "").upper()
            if sym.endswith(suffix):
                return sym
    for r in results:
        sym = (r.get("symbol") or "").upper()
        if sym and all(x not in sym for x in ["/", " ", ":"]):
            return sym
    return (results[0].get("symbol") or "").upper()

def lookup_ticker_for_name(name: str, exchange_hint: Optional[str] = None) -> Optional[str]:
    name = (name or "").strip()
    if not name: return None
    results = _finnhub_lookup_symbol(name)
    sym = _pick_symbol_from_lookup(results, exchange_hint)
    if not sym: return None
    if sym.endswith(".NS"): return sym[:-3]
    if sym.endswith(".BO"): return sym[:-3]
    return sym

def ensure_min_ohlc(final_output: Dict[str, Any]) -> None:
    """
    Ensure OHLC/price_range exist. If the exact 'date' has no row (weekend/holiday),
    backfill from the most recent available trading day within the last 14 days.
    """
    tkr = final_output.get("ticker") or final_output.get("company_ticker")
    if not tkr: return
    if final_output.get("ohlc") and final_output.get("price_range"): return

    exchange = final_output.get("exchange")
    yf_tkr = _yf_symbol(tkr, exchange)
    day = final_output.get("date") or get_curday()

    try:
        end_dt = datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)
        start_dt = end_dt - timedelta(days=14)
        df = yf.download(
            yf_tkr,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            progress=False
        )
        if len(df):
            row = df.iloc[-1]
            ohlc = {
                "O": float(row.get("Open", float("nan"))),
                "H": float(row.get("High", float("nan"))),
                "L": float(row.get("Low", float("nan"))),
                "C": float(row.get("Close", float("nan"))),
                "V": float(row.get("Volume", float("nan"))),
            }
            final_output.setdefault("ohlc", ohlc)
            final_output.setdefault("price_range", [ohlc["L"], ohlc["H"]])
        else:
            print(f"[ensure_min_ohlc] No rows for {yf_tkr} in the last 14 days ending {day}")
    except Exception as e:
        print(f"[ensure_min_ohlc] skipped: {e}")

def parse_query(q: str) -> Tuple[Optional[str], Optional[str]]:
    if not q: return None, None
    m_sym = re.search(r"\(([A-Z.\-]{1,10})\)", q.upper())
    if m_sym: ticker = m_sym.group(1)
    else:
        tickers = re.findall(r"\b[A-Z]{1,6}\b", q.upper())
        ticker = tickers[-1] if tickers else None
    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", q)
    return ticker, (m.group(1) if m else None)

def _clean_item(n: Dict[str, Any]) -> Optional[Dict[str, str]]:
    if not n: return None
    if str(n.get("summary","")).startswith("Looking for stock market analysis"): return None
    ts = n.get("datetime") or n.get("time") or n.get("publishedTime")
    try:
        dt = datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S") if ts else ""
    except Exception:
        dt = ""
    return {
        "date": dt,
        "headline": n.get("headline") or n.get("title") or "",
        "summary": n.get("summary") or n.get("description") or "",
        "source": n.get("source") or n.get("site") or "",
        "url": n.get("url") or n.get("link") or ""
    }

def _dedupe_news(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen, out = set(), []
    for it in items:
        key = it.get("url") or (it.get("headline"), it.get("date"))
        if key in seen: continue
        seen.add(key); out.append(it)
    return out

def get_company_news(symbol: str, start_date: str, end_date: str) -> List[Dict[str, str]]:
    try:
        raw = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
    except Exception as e:
        print(f"[news company] {e}"); raw = []
    items = []
    for n in raw or []:
        c = _clean_item(n)
        if c: items.append(c)
    return items

def news_for_window(symbol: str, anchor_day: str, weeks: int = 1) -> List[Dict[str, str]]:
    if not symbol: return []
    try:
        end_dt = datetime.strptime(anchor_day, "%Y-%m-%d")
    except Exception:
        end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=7*weeks)
    items = get_company_news(symbol, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    if items: return _dedupe_news(items)
    start_30 = end_dt - timedelta(days=30)
    items = get_company_news(symbol, start_30.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    return _dedupe_news(items)

def simple_sentiment_from_patterns(preds: List[Dict[str, Any]]) -> str:
    if not preds: return "Neutral"
    bullish = {"morning_star_rise","hammer","bullish_engulfing","ascending_triangle","golden_cross"}
    bearish = {"evening_star_fall","shooting_star","bearish_engulfing","descending_triangle","death_cross"}
    score = 0
    for p in preds:
        cls = str(p.get("class","")).lower()
        if any(b in cls for b in bullish): score += 1
        if any(b in cls for b in bearish): score -= 1
    return "Positive" if score>0 else "Negative" if score<0 else "Neutral"

# ── Local generation
def _truncate_at_stops(text: str, stops: List[str]) -> str:
    if not stops: return text
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1: cut = min(cut, i)
    return text[:cut]

def generate_text(prompt: str, max_new_tokens: int = 160, temperature: float = 0.2, top_p: float = 0.95, stop: Optional[List[str]] = None) -> str:
    stop = stop or []
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=120,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            streamer=None,
        )
    full = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    generated = full[len(prompt_text):]
    return _truncate_at_stops(generated, stop).strip()

# ── Summaries
def summarize_image(final_output: Dict[str, Any], sentiment: str, header: str = "Image Summary") -> str:
    tkr = final_output.get("ticker") or final_output.get("company_ticker") or "—"
    name = (final_output.get("company_name") or "").rstrip(":") or (tkr if tkr!="—" else "Unknown")
    exch = final_output.get("exchange") or "—"
    pr = final_output.get("price_range") or [None, None]
    ohlc = final_output.get("ohlc") or {}
    preds = final_output.get("predictions", [])
    patterns = ", ".join([p.get("class","") for p in preds]) if preds else "None"
    day = final_output.get("date") or get_curday()
    o, h, l, c, v = (ohlc.get("O","—"), ohlc.get("H","—"), ohlc.get("L","—"), ohlc.get("C","—"), ohlc.get("V","—"))
    pct = "—"
    try:
        if pr[0] and pr[1] and pr[0] not in (0,"—"): pct = f"{((pr[1]-pr[0])/pr[0]*100):.1f}%"
    except Exception:
        pass
    lines = [
        f"Analysis for **{name} ({tkr})** as of **{day}**{'' if exch=='—' else f' on **{exch}**'}.",
        f"Observed price range: **{pr[0]} – {pr[1]}** (≈ **{pct}** move).",
        f"OHLC snapshot: **O {o}**, **H {h}**, **L {l}**, **C {c}**, **V {v}**.",
        f"Detected patterns: **{patterns}**.",
        f"Overall sentiment: **{sentiment}**."
    ]
    return f"### {header}\n" + "\n".join(f"- {s}" for s in lines)

def llm_brief_summary(final_output: Dict[str, Any], sentiment: str, header: str = "Image Summary") -> str:
    ctx = {
        "ticker": final_output.get("ticker") or final_output.get("company_ticker"),
        "company_name": final_output.get("company_name"),
        "exchange": final_output.get("exchange"),
        "price_range": final_output.get("price_range"),
        "ohlc": final_output.get("ohlc"),
        "predictions": final_output.get("predictions"),
        "pattern_sentiment": sentiment,
    }
    sys_msg = ("You are a market analyst. Using ONLY the JSON, write 4–6 markdown bullets, full sentences for a non-technical reader. Avoid label:value.")
    user_msg = f"[JSON]\n{json.dumps(ctx, indent=2)}\n\n{OUTPUT_BEGIN}\n- Sentence...\n- Sentence...\n{OUTPUT_END}"
    prompt = f"{B_SYS}{sys_msg}{E_SYS}{B_INST}{user_msg}{E_INST}"
    raw = generate_text(prompt, max_new_tokens=200, temperature=0.0, top_p=1.0, stop=[OUTPUT_END])
    summary = _between_markers(raw).strip()
    lines = [ln for ln in summary.splitlines() if ln.strip().startswith("- ")]
    if len(lines) < 4:
        return summarize_image(final_output, sentiment, header=header)
    return f"### {header}\n" + "\n".join(lines)


# def build_llm_prompt(final_output: Dict[str, Any], news_snippets: List[Dict[str, str]]):
#     ctx = {
#         "ticker": final_output.get("ticker") or final_output.get("company_ticker"),
#         "company_name": final_output.get("company_name"),
#         "exchange": final_output.get("exchange"),
#         "ohlc": final_output.get("ohlc"),
#         "price_range": final_output.get("price_range"),
#         "predictions": final_output.get("predictions"),
#         "date": final_output.get("date"),
#     }
#     news_text = "\n".join(f"- {n.get('date','')} | {n.get('headline','')} :: {n.get('summary','')[:200]}..." for n in (news_snippets or [])[:6]) or "No recent news"
#     system = ("Use ONLY JSON and NEWS. All commentary must be about JSON.ticker. "
#               "Write exactly three markdown sections: Positive Developments, Potential Concerns, Forecast & Analysis.")
#     user = f"[JSON]\n{json.dumps(ctx, indent=2)}\n\n[NEWS]\n{news_text}\n\n{OUTPUT_BEGIN}\n<Your answer here>\n{OUTPUT_END}"
#     return f"{B_SYS}{system}{E_SYS}{B_INST}{user}{E_INST}"

def build_llm_prompt(final_output: Dict[str, Any], news_snippets: List[Dict[str, str]]):
    ctx = {
        "ticker": final_output.get("ticker") or final_output.get("company_ticker"),
        "company_name": final_output.get("company_name"),
        "exchange": final_output.get("exchange"),
        "ohlc": final_output.get("ohlc"),
        "price_range": final_output.get("price_range"),
        "predictions": final_output.get("predictions"),
        "date": final_output.get("date"),
    }
    news_text = "\n".join(
        f"- {n.get('date','')} | {n.get('headline','')} :: {n.get('summary','')[:200]}..."
        for n in (news_snippets or [])[:6]
    ) or "No recent news"

    system = (
        "Use ONLY JSON and NEWS. All commentary must be about JSON.ticker.\n"
        "Write exactly three markdown sections with these exact headings:\n"
        "1) ### Positive Developments\n"
        "2) ### Potential Concerns\n"
        "3) ### Forecast & Analysis\n"
        "Keep each section concise and actionable."
    )

    # Give it a skeleton to fill so it’s less likely to omit the 3rd section
    user = (
        f"[JSON]\n{json.dumps(ctx, indent=2)}\n\n[NEWS]\n{news_text}\n\n"
        f"{OUTPUT_BEGIN}\n"
        "### Positive Developments\n- \n- \n\n"
        "### Potential Concerns\n- \n- \n\n"
        "### Forecast & Analysis\n- \n- \n"
        f"{OUTPUT_END}"
    )
    return f"{B_SYS}{system}{E_SYS}{B_INST}{user}{E_INST}"


# def llm_forecast(final_output: Dict[str, Any], news_items: List[Dict[str, str]]) -> str:
#     try:
#         raw = generate_text(
#             build_llm_prompt(final_output, news_items),
#             max_new_tokens=320,
#             temperature=0.6,
#             top_p=0.9,
#             stop=[OUTPUT_END]
#         )
#         ans = _between_markers(raw).strip()
#         if ans:
#             return ans
#     except Exception as e:
#         print(f"[llm_forecast] generation error: {e}")

#     # Guaranteed fallback
#     tkr = final_output.get("ticker") or final_output.get("company_ticker") or "—"
#     pr = final_output.get("price_range") or [None, None]
#     return (
#         "### Positive Developments\n"
#         "- Liquidity appears stable; no acute stress flags in recent trading.\n\n"
#         "### Potential Concerns\n"
#         "- Pattern signals are weak/ambiguous; conviction limited without catalysts.\n\n"
#         f"### Forecast & Analysis\n"
#         f"- Expect range-bound move around **{pr[0]}–{pr[1]}** unless **{tkr}** breaks recent extremes.\n"
#     )

def llm_forecast(final_output: Dict[str, Any], news_items: List[Dict[str, str]]) -> str:
    try:
        raw = generate_text(
            build_llm_prompt(final_output, news_items),
            max_new_tokens=420,   # a bit larger to avoid truncation
            temperature=0.6,
            top_p=0.9,
            stop=[OUTPUT_END]
        )
        ans = _between_markers(raw).strip()
        if ans:
            # Ensure the section exists even if the model skipped it
            return _ensure_forecast_section(ans, final_output)
    except Exception as e:
        _dbg(f"[llm_forecast] generation error: {e}")

    # Guaranteed fallback if model failed or returned empty
    tkr = final_output.get("ticker") or final_output.get("company_ticker") or "—"
    pr = final_output.get("price_range") or [None, None]
    lo, hi = pr if isinstance(pr, (list, tuple)) and len(pr) == 2 else (None, None)
    return (
        "### Positive Developments\n"
        "- Liquidity appears stable; no acute stress flags in recent trading.\n\n"
        "### Potential Concerns\n"
        "- Pattern signals are weak/ambiguous; conviction limited without catalysts.\n\n"
        "### Forecast & Analysis\n"
        f"- Expect range-bound move around **{lo}–{hi}** unless **{tkr}** breaks recent extremes.\n"
    )


# ── Image / Text builders
def ensure_final_output_from_image(image_path: str) -> Dict[str, Any]:
    if not os.path.exists(YOLO_WEIGHTS):
        raise gr.Error(f"YOLO weights not found at: {YOLO_WEIGHTS}")

    metadata = StockChartMetadataExtractor(image_path).extract_metadata()
    predictor = StockChartTrendPredictor(YOLO_WEIGHTS)
    try: predictor.model.to("cuda")
    except Exception: pass

    preds, img = predictor.predict(image_path)
    final = predictor.save_predictions_to_json(preds, "final_output.json", img, metadata) or {}

    # Normalize mandatory fields
    final.setdefault("predictions", preds or [])
    final.setdefault("ohlc", {})
    final.setdefault("price_range", [None, None])

    # Normalize/force date
    img_date = final.get("date") or metadata.get("date")
    date_str = None
    if isinstance(img_date, str):
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                date_str = datetime.strptime(img_date, fmt).strftime("%Y-%m-%d")
                break
            except Exception:
                pass
    final["date"] = date_str or get_curday()

    # Ensure ticker (AAPL case included)
    ticker = final.get("ticker") or final.get("company_ticker")
    exchange_hint = final.get("exchange") or metadata.get("exchange")
    company_name = final.get("company_name") or metadata.get("company_name") or ""

    if not ticker:
        title = (final.get("company_name") or "") + " " + (final.get("title") or metadata.get("title") or "")
        m = re.search(r"\(([A-Z.\-]{1,10})\)", title.upper())
        if m:
            ticker = m.group(1)
    if not ticker and company_name:
        ticker = lookup_ticker_for_name(company_name, exchange_hint)

    if ticker:
        final["ticker"] = ticker.strip().upper()
    if exchange_hint:
        final["exchange"] = (_normalize_exchange_hint(exchange_hint) or "").upper() or final.get("exchange")

    # Ensure OHLC/price_range (14-day backfill)
    ensure_min_ohlc(final)

    with open("final_output.json","w",encoding="utf-8") as f: json.dump(final,f,indent=2,ensure_ascii=False)
    return final

def ensure_final_output_from_text(query: str) -> Dict[str, Any]:
    ticker, day = parse_query(query)
    day = day or get_curday()
    inferred_exchange = None
    if ticker:
        tU = ticker.upper()
        if tU.endswith(".NS"): inferred_exchange = "NSE"
        elif tU.endswith(".BO"): inferred_exchange = "BSE"
    final = {"ticker": ticker, "date": day, "source": "text_query", "predictions": [], "sessions": [], "price_range": [None,None], "ohlc": {}, "exchange": inferred_exchange, "company_name": None}
    try:
        if ticker:
            yf_tkr = _yf_symbol(ticker, inferred_exchange)
            # Try exact day first; if empty, backfill within 14 days (re-use ensure_min_ohlc)
            end_day = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            df = yf.download(yf_tkr, start=day, end=end_day, progress=False)
            if len(df):
                row = df.iloc[0]
                final["ohlc"] = {"O": float(row.get("Open", float("nan"))), "H": float(row.get("High", float("nan"))), "L": float(row.get("Low", float("nan"))), "C": float(row.get("Close", float("nan"))), "V": float(row.get("Volume", float("nan"))) }
                final["price_range"] = [float(row.get("Low")), float(row.get("High"))]
            else:
                ensure_min_ohlc(final)
            try:
                tk = yf.Ticker(yf_tkr)
                info = {}
                try: info = tk.get_info()
                except Exception: info = {}
                if not final["company_name"]: final["company_name"] = info.get("shortName") or info.get("longName")
                if not final["exchange"]:
                    ex = info.get("fullExchangeName") or info.get("exchange")
                    if isinstance(ex,str) and ex.strip(): final["exchange"] = ex
            except Exception as e2:
                print(f"[text enrich] skipped: {e2}")
    except Exception as e:
        print(f"[yfinance enrich] skipped: {e}")
    with open("final_output.json","w",encoding="utf-8") as f: json.dump(final,f,indent=2,ensure_ascii=False)
    return final

# ── RAG helpers (financial docs)
def _make_unique_index_dir(file_name: str) -> Path:
    h = hashlib.sha1(file_name.encode("utf-8")).hexdigest()[:12]
    d = Path("./faiss_indexes") / f"idx_{h}"
    d.mkdir(parents=True, exist_ok=True)
    return d

# def prepare_rag_session_for_file(file_path: str):
#     filename = Path(file_path).name
#     with open(file_path,"rb") as f: file_bytes = f.read()
#     frs.INDEX_DIR = _make_unique_index_dir(filename)
#     session = frs.prepare_session_any(file_bytes, filename)
#     ctx, meta = frs.get_summary_context(session, max_chunks=8)
#     sys_msg = ("You are a senior equity analyst. Provide a concise executive summary of the document "
#                "covering: company, filing/report type, period/fiscal year, revenue/earnings highlights, "
#                "segment/geographic notes, liquidity & capital resources, key risks, guidance/outlook, dividends/buybacks. Use bullets.")
#     head = f"Meta: company={meta.company}, ticker={meta.ticker}, FY={meta.fiscal_year}, period={meta.period}"
#     prompt = f"{B_SYS}{sys_msg}{E_SYS}{B_INST}{head}\n\nContext:\n{ctx}\n\nSummary:{E_INST}"
#     summary = generate_text(prompt, max_new_tokens=450, temperature=0.2)
#     return session, meta, f"**Executive Summary — {meta.title or filename}**\n\n{summary}"

# def retrieve_across_sessions(question: str, sessions: List[frs.RAGSession], top_k: int = None):
#     if not sessions: return {"context":"", "sources":[]}
#     top_k = top_k or frs.TOP_K
#     ctx_parts, sources = [], []
#     for s in sessions:
#         r = frs.retrieve(s, question, k=top_k)
#         ctx_parts.append(r.get("context",""))
#         sources.extend(r.get("sources",[]))
#     return {"context":"\n\n".join(ctx_parts), "sources":sources}

# def format_sources_for_display(sources):
#     if not sources: return ""
#     return "\n".join(f"- {Path(s.get('source','unknown')).name} · chunk {s.get('chunk_id','?')}" for s in sources)

# ── Core turn handlers
# def run_market_pipeline(final_output: Dict[str, Any], do_news: bool):
#     ticker = final_output.get("ticker") or final_output.get("company_ticker")
#     if not ticker: raise gr.Error("Ticker not found.")
#     anchor_date = final_output.get("date") or get_curday()
#     ensure_min_ohlc(final_output)
#     sentiment = simple_sentiment_from_patterns(final_output.get("predictions", []))
#     try: summary_md = llm_brief_summary(final_output, sentiment)
#     except Exception as e:
#         print("brief summary fallback:", e)
#         summary_md = summarize_image(final_output, sentiment)
#     news_items = news_for_window(ticker, anchor_date, weeks=1) if do_news else []
#     forecast_md = llm_forecast(final_output, news_items)
#     json_path = os.path.abspath("final_output.json")
#     with open(json_path,"w",encoding="utf-8") as f: json.dump(final_output,f,indent=2,ensure_ascii=False)
#     final_json_str = json.dumps(final_output, indent=2, ensure_ascii=False)
#     return summary_md, sentiment, forecast_md, final_json_str, pd.DataFrame(news_items) if news_items else pd.DataFrame([{"info":"News omitted or none found."}]), json_path

def run_market_pipeline(final_output: Dict[str, Any], do_news: bool):
    """
    Robust market pipeline: never throws out to Gradio.
    Returns summary_md, sentiment, forecast_md, final_json_str, news_df, json_path
    """
    try:
        t_symbol = final_output.get("ticker") or final_output.get("company_ticker")
        if not t_symbol:
            raise gr.Error("Ticker not found in input. For images, make sure the OCR/metadata contains a ticker, or add it in the title as (TICKER).")

        anchor_date = final_output.get("date") or get_curday()
        header = "Market Summary" if final_output.get("source") == "text_query" else "Image Summary"

        # Ensure OHLC (with weekend/holiday backfill)
        _dbg(f"pipeline start: ticker={t_symbol}, date={anchor_date}, ex={final_output.get('exchange')}")
        ensure_min_ohlc(final_output)
        _dbg(f"ohlc: {final_output.get('ohlc')}, price_range: {final_output.get('price_range')}")

        # Sentiment from patterns (safe)
        sentiment = simple_sentiment_from_patterns(final_output.get("predictions", []))

        # Summary (LLM w/ fallback)
        try:
            summary_md = llm_brief_summary(final_output, sentiment, header=header)
        except Exception as e_sum:
            _dbg(f"llm_brief_summary error: {e_sum}")
            summary_md = summarize_image(final_output, sentiment, header=header)

        # News (optional)
        news_items = []
        if do_news:
            try:
                news_items = news_for_window(t_symbol, anchor_date, weeks=1)
            except Exception as e_news:
                _dbg(f"news_for_window error: {e_news}")
                news_items = []

        _dbg(f"news_count={len(news_items)}")

        # Forecast (LLM w/ guaranteed fallback)
        forecast_md = llm_forecast(final_output, news_items)

        # Persist JSON
        json_path = os.path.abspath("final_output.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        final_json_str = json.dumps(final_output, indent=2, ensure_ascii=False)

        news_df = pd.DataFrame(news_items) if news_items else pd.DataFrame([{"info": "News omitted or none found."}])
        return summary_md, sentiment, forecast_md, final_json_str, news_df, json_path

    except Exception as e:
        # Catch-all safeguard so the UI never crashes
        err = f"{type(e).__name__}: {e}"
        _dbg(f"run_market_pipeline failed: {err}\n{traceback.format_exc()}")

        # Minimal safe payload back to UI
        fallback_summary = "### Market Summary\n- An error occurred while generating the analysis."
        fallback_sentiment = "Unknown"
        fallback_forecast = (
            "### Forecast & Analysis\n"
            "- Unable to compute forecast due to an internal error. Please check the console logs for details."
        )
        try:
            # try to preserve whatever JSON we had
            json_path = os.path.abspath("final_output.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(final_output or {"error": err}, f, indent=2, ensure_ascii=False)
            final_json_str = json.dumps(final_output or {"error": err}, indent=2, ensure_ascii=False)
        except Exception:
            json_path = None
            final_json_str = json.dumps({"error": err}, indent=2)

        news_df = pd.DataFrame([{"error": err}])
        return fallback_summary, fallback_sentiment, fallback_forecast, final_json_str, news_df, json_path


def handle_user_turn(history_pairs, user_text: str, user_files: Optional[List[str]], do_news: bool, rag_sessions: list):
    history_pairs = history_pairs or []
    file_paths = []
    if user_files:
        for f in user_files:
            if isinstance(f,str): file_paths.append(f)
            elif isinstance(f,dict): file_paths.append(f.get("path") or f.get("name"))
            else: file_paths.append(str(f))
    file_paths = [p for p in file_paths if p]

    user_display = (user_text or "").strip() or "(no text)"
    if file_paths: user_display += "\n[attachments: " + ", ".join(Path(p).name for p in file_paths) + "]"
    history_pairs = history_pairs + [(user_display, "")]

    assistant_sections = []
    last_summary = last_sent = last_forecast = last_json_str = None
    last_news_df, last_json_path = pd.DataFrame(), None

    for p in file_paths:
        ext = Path(p).suffix.lower()
        if ext in IMG_EXT:
            final_output = ensure_final_output_from_image(p)
            s_md, sent, f_md, j_str, n_df, j_path = run_market_pipeline(final_output, do_news)
            assistant_sections.append(
                f"**📈 Chart analyzed:** {Path(p).name}\n\n{s_md}\n\n---\n**Sentiment:** `{sent}`\n\n### LLM Forecast\n{f_md}\n\n"
                f"<sub>Note: date used = **{final_output.get('date','?')}**, ticker = **{final_output.get('ticker','?')}**, exchange = **{final_output.get('exchange','—')}**</sub>"
            )
            last_summary, last_sent, last_forecast = s_md, sent, f_md
            last_json_str, last_news_df, last_json_path = j_str, n_df, j_path

        elif ext in DOC_EXT:
            # session, meta, summary_block = prepare_rag_session_for_file(p)
            # rag_sessions.append(session)
            # possible_ticker = getattr(meta, "ticker", None) or parse_query(meta.title or "")[0] or parse_query(Path(p).stem)[0]
            # if possible_ticker:
            #     ticker_output = {"ticker": possible_ticker, "date": get_curday(), "predictions": [], "price_range":[None,None], "ohlc":{}, "exchange": None, "company_name": getattr(meta,"company",None)}
            #     s_md, sent, f_md, j_str, n_df, j_path = run_market_pipeline(ticker_output, do_news)
            #     assistant_sections.append(
            #         f"**📄 Document processed:** {Path(p).name}\n\n{summary_block}\n\n---\n**Ticker context:** `{possible_ticker}`\n\n"
            #         f"{s_md}\n\n**Sentiment:** `{sent}`\n\n### LLM Forecast\n{f_md}"
            #     )
            #     last_summary, last_sent, last_forecast = s_md, sent, f_md
            #     last_json_str, last_news_df, last_json_path = j_str, n_df, j_path
            # else:
                assistant_sections.append(f"**📄 Document processed:** {Path(p).name}\n\n{summary_block}\n\n_(No ticker detected in doc meta/title; ask a question with the ticker to analyze market context.)_")

    if user_text and not file_paths:
        ticker, _ = parse_query(user_text)
        if ticker:
            final_output = ensure_final_output_from_text(user_text)
            s_md, sent, f_md, j_str, n_df, j_path = run_market_pipeline(final_output, do_news)
            assistant_sections.append(f"**📊 Market Query:** {user_text}\n\n{s_md}\n\n---\n**Sentiment:** `{sent}`\n\n### LLM Forecast\n{f_md}")
            last_summary, last_sent, last_forecast = s_md, sent, f_md
            last_json_str, last_news_df, last_json_path = j_str, n_df, j_path
        elif rag_sessions:
            ret = retrieve_across_sessions(user_text, rag_sessions, top_k=frs.TOP_K)
            ctx, sources = ret["context"], ret["sources"]
            qa_prompt = f"You are a diligent financial analyst. Use ONLY the provided context.\n\nQuestion:\n{user_text}\n\nContext:\n{ctx}\n\nAnswer:"
            ans = generate_text(qa_prompt, max_new_tokens=420, temperature=0.2)
            srcs = format_sources_for_display(sources)
            if srcs: ans += f"\n\n---\n{srcs}"
            assistant_sections.append(f"**❓ Doc-grounded answer**\n\n{ans}")
        else:
            assistant_sections.append("⚠️ Provide a ticker/date or upload a chart/document.")

    assistant_md = "\n\n".join(assistant_sections) if assistant_sections else "No output."
    history_pairs[-1] = (history_pairs[-1][0], assistant_md)
    return history_pairs, last_summary or "", last_sent or "", last_forecast or "", last_json_str or "", last_news_df, last_json_path

# ── UI
with gr.Blocks(title="FinGPT-M — Unified Chat") as demo:
    gr.HTML("""
    <style>
      .compact-container {max-width: 980px; margin: 0 auto;}
      .gradio-container {background: #0b0f17;}
      .gr-chatbot {border-radius: 14px;}
      .message-wrap .message {border-radius: 16px; padding: 10px 14px;}
      .btn-row {display:flex; gap:12px; flex-wrap:wrap;}
      .shadow-card {background: #0f1623; border: 1px solid #1f2a3a; border-radius: 16px; padding: 12px;}
    </style>
    """)

    with gr.Row():
        with gr.Column(elem_classes=["compact-container"], scale=8):
            gr.Markdown("## FinGPT-M — Unified Chat\nAsk with a **ticker** or **date** (e.g., TSLA (2025-06-04)), or drop **charts** and **financial documents** together.")

            chat = gr.Chatbot(label=None, height=560, show_copy_button=True, bubble_full_width=False)

            mm = gr.MultimodalTextbox(
                placeholder="Ask: “Analyze TSLA (2025-06-04)”, or drop charts + PDFs…",
                show_label=False,
                file_types=list(IMG_EXT | DOC_EXT),
                autofocus=True,
                submit_btn=True,
                # max_files=8,
            )

            summary_out = gr.Markdown(visible=False)
            sentiment_out = gr.Textbox(visible=False)
            forecast_out = gr.Markdown(visible=False)
            json_out = gr.Code(language="json", label="", visible=False)
            news_out = gr.Dataframe(visible=False)
            json_download = gr.DownloadButton(label="Download final_output.json", visible=False)

        with gr.Column(min_width=360, scale=4):
            with gr.Group():
                gr.Markdown("### Tools")
                with gr.Column(elem_classes=["shadow-card"]):
                    gr.Markdown("**final_output.json** (last market analysis)")
                    with gr.Accordion("Preview", open=False):
                        json_preview = gr.Code(language="json", label="", interactive=False)
                    drow = gr.Row(elem_classes=["btn-row"])
                    with drow:
                        json_download_side = gr.DownloadButton("Download JSON")
                with gr.Column(elem_classes=["shadow-card"]):
                    gr.Markdown("**Recent News (toggle)**")
                    news_toggle = gr.Checkbox(label="Pull recent news (Finnhub)", value=False)
                    news_table = gr.Dataframe(wrap=True)
                with gr.Column(elem_classes=["shadow-card"]):
                    gr.Markdown("**Session Controls**")
                    clear_btn = gr.Button("🔄 Reset chat & sessions", variant="secondary")

    chat_state = gr.State([])
    rag_sessions_state = gr.State([])

    def on_mm_submit_pairs(history_pairs, rag_sessions, data, do_news):
        user_text = (data or {}).get("text") or ""
        user_files = (data or {}).get("files") or []
        return handle_user_turn(history_pairs, user_text, user_files, do_news, rag_sessions)

    mm.submit(
        fn=on_mm_submit_pairs,
        inputs=[chat_state, rag_sessions_state, mm, news_toggle],
        outputs=[chat, summary_out, sentiment_out, forecast_out, json_out, news_out, json_download],
    ).then(
        fn=lambda jo, df, jp: (gr.update(value=jo), df, jp),
        inputs=[json_out, news_out, json_download],
        outputs=[json_preview, news_table, json_download_side],
    ).then(
        fn=lambda h: h,
        inputs=chat,
        outputs=chat_state,
    )

    def on_clear():
        return [], [], "", "", "", pd.DataFrame(), None

    clear_btn.click(
        fn=on_clear,
        inputs=[],
        outputs=[chat, rag_sessions_state, summary_out, sentiment_out, forecast_out, news_out, json_download],
    )

if __name__ == "__main__":
    if not os.path.exists(YOLO_WEIGHTS):
        raise FileNotFoundError(f"YOLO weights not found at: {YOLO_WEIGHTS}")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    # demo.queue(concurrency_count=1, max_size=16)
    demo.launch(share=True, debug=True)
