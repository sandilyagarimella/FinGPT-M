import os
import re
import json
import time
import traceback
from typing import Dict, Any, Optional, Tuple, List

import sys
# >>> adjust to your local repo layout
sys.path.insert(0, r"/content/FinGPT-M/fingpt/stock_chart_trends_analysis")

import torch
import gradio as gr
import pandas as pd
import yfinance as yf
import finnhub
from dotenv import load_dotenv
from datetime import date, datetime, timedelta

# ─────────────────────────────────────────────────────────────
# NEW: Local LLM (transformers + peft)
# ─────────────────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig
from peft import PeftModel

# ─────────────────────────────────────────────────────────────
# Paths & env
# ─────────────────────────────────────────────────────────────
YOLO_WEIGHTS = r"/content/FinGPT-M/fingpt/stock_chart_trends_analysis/best.pt"
os.environ["YOLO_MODEL_PATH"] = YOLO_WEIGHTS
os.environ["ULTRALYTICS_VERBOSE"] = "False"

load_dotenv(override=True)

from google.colab import userdata

HF_TOKEN = userdata.get("HF_TOKEN")
FINNHUB_API_KEY = userdata.get("FINNHUB_API_KEY")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set. Put it in your environment or .env")
if not FINNHUB_API_KEY:
    raise RuntimeError("FINNHUB_API_KEY not set. Put it in your environment or .env")

# ── Create the client with the **string** key
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)


# ─────────────────────────────────────────────────────────────
# Imports from your pipeline
# ─────────────────────────────────────────────────────────────
from StockChart_Trend_Prediction import StockChartTrendPredictor, StockChartMetadataExtractor

# ─────────────────────────────────────────────────────────────
# Finnhub client (for news)
# ─────────────────────────────────────────────────────────────
# finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)

# ─────────────────────────────────────────────────────────────
# LLM (Local): Llama-2-7B-Chat + FinGPT LoRA
# ─────────────────────────────────────────────────────────────
access_token = HF_TOKEN  # Llama-2 base is gated; uses HF token

LLAMA_BASE = "meta-llama/Llama-2-7b-chat-hf"
LORA_ADAPTER = "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"

CACHE_DIR = r"E:/FinGPT/llama_cache"
OFFLOAD_DIR = r"E:/FinGPT/offload"

# Safe dtype/device selection
if torch.cuda.is_available():
    dtype = torch.bfloat16  # safer than fp16 for many CUDA setups
    device_map = "auto"     # let accelerate map layers to GPU
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=False, # Set to False to disable 4-bit quantization
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
else:
    dtype = torch.float32   # CPU needs fp32 (fp16 on CPU will crash)
    device_map = "cpu"
    bnb_config = None # No quantization on CPU

print("• Loading base model (first run can take a while)…")
base_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_BASE,
    token=access_token,
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
    device_map=device_map,
    torch_dtype=dtype,
    offload_folder=OFFLOAD_DIR,
    quantization_config=bnb_config if torch.cuda.is_available() else None, # Apply quantization config if GPU is available
)

print("• Applying LoRA adapter…")
model = PeftModel.from_pretrained(
    base_model,
    LORA_ADAPTER,
    offload_folder=OFFLOAD_DIR,
    cache_dir=CACHE_DIR,
)
model = model.eval()

print("• Loading tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(
    LLAMA_BASE,
    token=access_token,
    cache_dir=CACHE_DIR,
    use_fast=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# optional console streamer (disabled by default in generate_text)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Llama-2 chat tokens + our fence markers
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

OUTPUT_BEGIN = "[OUTPUT_BEGIN]"
OUTPUT_END = "[OUTPUT_END]"

def _between_markers(text: str) -> str:
    if OUTPUT_BEGIN in text:
        text = text.split(OUTPUT_BEGIN, 1)[1]
    if OUTPUT_END in text:
        text = text.split(OUTPUT_END, 1)[0]
    return text.strip()

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _yf_symbol(ticker: str, exchange: Optional[str]) -> str:
    if not ticker: return ticker
    t = ticker.upper().replace(" ", "")
    ex = (exchange or "").upper()
    if ex in {"NSE","NSEI","INDIA"} and not t.endswith(".NS"): return f"{t}.NS"
    if ex == "BSE" and not t.endswith(".BO"): return f"{t}.BO"
    return t

def ensure_min_ohlc(final_output: Dict[str, Any]) -> None:
    """If OHLC/price_range are missing but we have a ticker, fetch 1 day from yfinance."""
    tkr = final_output.get("ticker") or final_output.get("company_ticker")
    if not tkr:
        return
    if final_output.get("ohlc") and final_output.get("price_range"):
        return

    exchange = final_output.get("exchange")
    yf_tkr = _yf_symbol(tkr, exchange)
    day = final_output.get("date") or get_curday()
    try:
        end_day = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.download(yf_tkr, start=day, end=end_day, progress=False)
        if len(df):
            row = df.iloc[0]
            ohlc = {
                "O": float(row["Open"]),
                "H": float(row["High"]),
                "L": float(row["Low"]),
                "C": float(row["Close"]),
                "V": float(row["Volume"]),
            }
            final_output.setdefault("ohlc", ohlc)
            final_output.setdefault("price_range", [ohlc["L"], ohlc["H"]])
    except Exception as e:
        print(f"[ensure_min_ohlc] yfinance fallback skipped: {e}")


def get_curday() -> str:
    return date.today().strftime("%Y-%m-%d")

def parse_query(q: str) -> Tuple[Optional[str], Optional[str]]:
    if not q:
        return None, None
    m_sym = re.search(r"\(([A-Z.\-]{1,10})\)", q.upper())
    if m_sym:
        ticker = m_sym.group(1)
    else:
        tickers = re.findall(r"\b[A-Z]{1,5}\b", q.upper())
        ticker = tickers[-1] if tickers else None
    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", q)
    day = m.group(1) if m else None
    return ticker, day

from finnhub import FinnhubAPIException

def _candidate_symbols_for_news(ticker: str, exchange: Optional[str]) -> List[str]:
    t = (ticker or "").upper().replace(" ", "")
    ex = (exchange or "").upper()
    cands = [t]
    # NSE/BSE common formats for Finnhub:
    if ex in {"NSE", "NSEI", "INDIA"} or t.endswith(".NS"):
        base = t.replace(".NS", "")
        cands = [f"{base}.NS", f"NSE:{base}", base] + cands
    if ex in {"BSE"} or t.endswith(".BO"):
        base = t.replace(".BO", "")
        cands = [f"{base}.BO", f"BSE:{base}", base] + cands
    # Dedup, keep order
    seen, ordered = set(), []
    for s in cands:
        if s and s not in seen:
            seen.add(s); ordered.append(s)
    return ordered

def get_company_news(symbol: str, start_date: str, end_date: str, exchange: Optional[str] = None) -> List[Dict[str, str]]:
    if not symbol:
        return []
    items = []
    for sym in _candidate_symbols_for_news(symbol, exchange):
        try:
            raw = finnhub_client.company_news(sym, _from=start_date, to=end_date)
            if raw:
                items = raw
                break
        except FinnhubAPIException as e:
            print(f"[WARN] Finnhub({sym}) failed: {e}")
            continue

    cleaned = [
        {
            "date": datetime.fromtimestamp(n["datetime"]).strftime("%Y-%m-%d %H:%M:%S"),
            "headline": n.get("headline", ""),
            "summary": n.get("summary", ""),
            "source": n.get("source", ""),
            "url": n.get("url", ""),
        }
        for n in (items or [])
        if not str(n.get("summary", "")).startswith("Looking for stock market analysis")
    ]
    # Top 8, dedup by headline
    seen, out = set(), []
    for r in cleaned:
        if r["headline"] not in seen:
            seen.add(r["headline"])
            out.append(r)
        if len(out) >= 8:
            break
    return out

def news_for_window(symbol: str, anchor_day: str, exchange: Optional[str], weeks: int = 1) -> List[Dict[str, str]]:
    try:
        end_dt = datetime.strptime(anchor_day, "%Y-%m-%d")
    except Exception:
        end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=7 * weeks)
    # Small guard: long weekends/no news
    return get_company_news(symbol, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), exchange)


def simple_sentiment_from_patterns(preds: List[Dict[str, Any]]) -> str:
    if not preds:
        return "Neutral"
    bullish = {"morning_star_rise", "hammer", "bullish_engulfing", "ascending_triangle", "golden_cross"}
    bearish = {"evening_star_fall", "shooting_star", "bearish_engulfing", "descending_triangle", "death_cross"}
    score = 0
    for p in preds:
        cls = str(p.get("class", "")).lower()
        if any(b in cls for b in bullish):
            score += 1
        if any(b in cls for b in bearish):
            score -= 1
    return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"

# ─────────────────────────────────────────────────────────────
# Local text generation (replaces old HTTP-based generate_text)
# ─────────────────────────────────────────────────────────────
def _truncate_at_stops(text: str, stops: List[str]) -> str:
    if not stops:
        return text
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]

def generate_text(
    prompt: str,
    max_new_tokens: int = 160,
    temperature: float = 0.2,
    top_p: float = 0.95,
    stop: Optional[List[str]] = None,
    timeout: int = 60,  # kept for API compat; unused locally
) -> str:
    stop = stop or []

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        streamer=None,  # set to streamer to print live to console
    )

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    full = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    generated = full[len(prompt_text):]
    generated = _truncate_at_stops(generated, stop)
    return generated.strip()

# Fallback deterministic summary
# Pattern explanation map
PATTERN_EXPLANATIONS = {
    "evening_star_fall": "a bearish reversal signal, often suggesting that an uptrend may be coming to an end.",
    "morning_star_rise": "a bullish reversal signal, often indicating that a downtrend may reverse upward.",
    "hammer": "a bullish reversal pattern, which can signal potential buying interest after a decline.",
    "shooting_star": "a bearish reversal pattern, which can indicate potential selling pressure at higher prices.",
    "bullish_engulfing": "a strong bullish reversal pattern where buying pressure overcomes prior selling.",
    "bearish_engulfing": "a strong bearish reversal pattern where selling pressure overcomes prior buying.",
    "ascending_triangle": "a bullish continuation pattern that often precedes a breakout to higher prices.",
    "descending_triangle": "a bearish continuation pattern that often precedes a breakdown to lower prices.",
    "golden_cross": "a bullish signal that occurs when the short-term moving average crosses above the long-term moving average.",
    "death_cross": "a bearish signal that occurs when the short-term moving average crosses below the long-term moving average."
}


def summarize_image(final_output: Dict[str, Any], sentiment: str) -> str:
    tkr = final_output.get("ticker") or final_output.get("company_ticker") or "—"
    name = (final_output.get("company_name") or "").rstrip(":") or "Unknown company"
    exch = final_output.get("exchange") or "—"
    pr = final_output.get("price_range") or [None, None]
    ohlc = final_output.get("ohlc") or {}
    preds = final_output.get("predictions", [])
    patterns_list = [p.get("class", "") for p in preds] if preds else []
    patterns_str = ", ".join(patterns_list) if patterns_list else "None"

    o, h, l, c, v = (
        ohlc.get("O", "—"), ohlc.get("H", "—"), ohlc.get("L", "—"),
        ohlc.get("C", "—"), ohlc.get("V", "—")
    )
    pct = f"{((pr[1] - pr[0]) / pr[0] * 100):.1f}%" if pr[0] and pr[1] and pr[0] else "—"

    # Reasoning for first detected pattern
    pattern_reason = ""
    if patterns_list and patterns_list[0] in PATTERN_EXPLANATIONS:
        pattern_reason = f" This pattern is {PATTERN_EXPLANATIONS[patterns_list[0]]}"

    # Sentiment reasoning
    sentiment_reason = ""
    if sentiment.lower() == "positive":
        sentiment_reason = " This suggests buying momentum or market optimism."
    elif sentiment.lower() == "negative":
        sentiment_reason = " This suggests selling pressure or market caution."
    elif sentiment.lower() == "neutral":
        sentiment_reason = " This indicates a balanced sentiment without strong directional bias."

    sentences = [
        f"The chart belongs to **{name} ({tkr})**, listed on **{exch}**.",
        f"The observed price range during the session was from **{pr[0]}** to **{pr[1]}**, representing a movement of about **{pct}**.",
        f"The OHLC values were **Open {o}**, **High {h}**, **Low {l}**, and **Close {c}**, with a traded volume of **{v}** shares.",
        f"Identified chart patterns: **{patterns_str}**.{pattern_reason}",
        f"Overall market sentiment based on detected patterns is assessed as **{sentiment}**.{sentiment_reason}"
    ]

    return "### Image Summary\n" + "\n".join(f"- {s}" for s in sentences)


def llm_brief_summary(final_output: Dict[str, Any], sentiment: str) -> str:
    ctx = {
        "ticker": final_output.get("ticker") or final_output.get("company_ticker"),
        "company_name": final_output.get("company_name"),
        "exchange": final_output.get("exchange"),
        "price_range": final_output.get("price_range"),
        "ohlc": final_output.get("ohlc"),
        "predictions": final_output.get("predictions"),
        "pattern_sentiment": sentiment,
    }

    sys_msg = (
        "You are a market analyst. Using ONLY the JSON, write 4–6 markdown bullet points "
        "where each bullet is a full sentence explaining the detail in plain language for a non-technical reader. "
        "Include reasoning for any detected chart patterns (what they usually indicate) and sentiment (what it implies). "
        "Avoid table-like label:value formats."
    )

    user_msg = f"[JSON]\n{json.dumps(ctx, indent=2)}\n\n{OUTPUT_BEGIN}\n- Sentence...\n- Sentence...\n{OUTPUT_END}"
    prompt = f"{B_SYS}{sys_msg}{E_SYS}{B_INST}{user_msg}{E_INST}"

    raw = generate_text(prompt, max_new_tokens=180, temperature=0.0, top_p=1.0, stop=[OUTPUT_END])
    summary = _between_markers(raw).strip()
    lines = [ln for ln in summary.splitlines() if ln.strip().startswith("- ")]

    # Deterministic fallback if the LLM under-delivers
    if len(lines) < 4:
        return summarize_image(final_output, sentiment)

    return "### Image Summary\n" + "\n".join(lines)



# ---------- Forecast helpers ----------

def build_llm_prompt(final_output: Dict[str, Any], news_snippets: List[Dict[str, str]], require_patterns: bool = False):
    """
    Forecast prompt that works with OR without patterns.
    `require_patterns` kept for compatibility; if False, we allow empty predictions.
    """
    ctx = {
        "ticker": final_output.get("ticker") or final_output.get("company_ticker"),
        "company_name": final_output.get("company_name"),
        "exchange": final_output.get("exchange"),
        "ohlc": final_output.get("ohlc"),
        "price_range": final_output.get("price_range"),
        "predictions": (final_output.get("predictions") or []) if not require_patterns else final_output.get("predictions"),
        "date": final_output.get("date"),
    }

    news_text = "\n".join(
        f"- {n.get('date','')} | {n.get('headline','')} :: {n.get('summary','')[:200]}..."
        for n in (news_snippets or [])[:6]
    ) or "No recent news"

    system = (
        "You are a cautious stock analyst. Use ONLY the JSON/NEWS provided. "
        "Write exactly three markdown sections in this order: "
        "Positive Developments, Potential Concerns, Forecast & Analysis. "
        "If no chart patterns exist, rely on ticker/company, OHLC and price range, and any news context. "
        "Be specific but avoid guarantees/targets. No extra sections or links."
    )

    user_fenced = (
        f"[JSON]\n{json.dumps(ctx, indent=2)}\n\n"
        f"[NEWS]\n{news_text}\n\n"
        f"{OUTPUT_BEGIN}\n<Your answer here>\n{OUTPUT_END}"
    )

    user_unfenced = (
        f"[JSON]\n{json.dumps(ctx, indent=2)}\n\n"
        f"[NEWS]\n{news_text}\n\n"
        "Write the three sections now."
    )

    return system, user_fenced, user_unfenced


def _baseline_forecast(final_output: Dict[str, Any]) -> str:
    tkr = final_output.get("ticker") or final_output.get("company_ticker") or "—"
    name = (final_output.get("company_name") or "").rstrip(":") or "the company"
    exch = final_output.get("exchange") or "—"
    pr = final_output.get("price_range") or [None, None]
    ohlc = final_output.get("ohlc") or {}
    o, h, l, c = ohlc.get("O"), ohlc.get("H"), ohlc.get("L"), ohlc.get("C")

    pct_txt = "a limited range"
    if pr[0] and pr[1] and pr[0] not in (0, "—"):
        try:
            pct = (pr[1] - pr[0]) / pr[0] * 100
            pct_txt = f"{abs(pct):.1f}% range"
        except Exception:
            pass

    return (
        "### Positive Developments\n"
        f"- **{name} ({tkr}, {exch})** traded between **{pr[0]}** and **{pr[1]}**, indicating {pct_txt}.\n"
        f"- The latest OHLC snapshot (**O {o}, H {h}, L {l}, C {c}**) provides nearby support/resistance.\n\n"
        "### Potential Concerns\n"
        "- No strong chart patterns were detected, reducing directional conviction.\n"
        "- Limited news context in this window; signals may rely more on technical levels.\n\n"
        "### Forecast & Analysis\n"
        "- Expect **range-bound** behavior unless price **breaks above the recent high** (bullish bias) or **falls below the recent low** (bearish bias).\n"
        "- Favor risk-managed entries around clear levels until stronger catalysts emerge.\n"
    )


def llm_forecast_strong(final_output: Dict[str, Any], news_items: List[Dict[str, str]]) -> str:
    # 1) fenced attempt
    system, user_fenced, user_unfenced = build_llm_prompt(final_output, news_items, require_patterns=False)
    prompt_fenced = f"{B_SYS}{system}{E_SYS}{B_INST}{user_fenced}{E_INST}"
    raw1 = generate_text(prompt_fenced, max_new_tokens=300, temperature=0.6, top_p=0.9, stop=[OUTPUT_END])
    ans1 = _between_markers(raw1).strip()
    if ans1:
        return ans1

    # 2) unfenced attempt
    prompt_unfenced = f"{B_SYS}{system}{E_SYS}{B_INST}{user_unfenced}{E_INST}"
    raw2 = generate_text(prompt_unfenced, max_new_tokens=300, temperature=0.6, top_p=0.9)
    ans2 = raw2.strip()
    if ans2:
        return ans2

    # 3) deterministic fallback
    return _baseline_forecast(final_output)


# ─────────────────────────────────────────────────────────────
# Build final_output.json
# ─────────────────────────────────────────────────────────────
def ensure_final_output_from_image(image_path: str) -> Dict[str, Any]:
    if not os.path.exists(YOLO_WEIGHTS):
        raise gr.Error(f"YOLO weights not found at: {YOLO_WEIGHTS}")
    print(f"Attempting to load image from: {image_path}")
    metadata_extractor = StockChartMetadataExtractor(image_path)
    metadata = metadata_extractor.extract_metadata()

    predictor = StockChartTrendPredictor(YOLO_WEIGHTS)
    try:
        predictor.model.to("cuda")
    except Exception:
        pass
    preds, img = predictor.predict(image_path)

    final = predictor.save_predictions_to_json(preds, "final_output.json", img, metadata)

    if not final.get("ticker") and not final.get("company_ticker"):
        title = (final.get("company_name") or "") + " " + (final.get("title") or "")
        m = re.search(r"\(([A-Z.\-]{1,10})\)", title.upper())
        if m:
            final["ticker"] = m.group(1)

    return final

def ensure_final_output_from_text(query: str) -> Dict[str, Any]:
    ticker, day = parse_query(query)
    final = {
        "ticker": ticker,
        "date": day or get_curday(),
        "source": "text_query",
        "predictions": [],
        "sessions": [],
        "price_range": [None, None],
        "ohlc": {},
        "exchange": None,
        "company_name": None,
    }
    try:
        if ticker:
            end_day = (datetime.strptime(final["date"], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            df = yf.download(ticker, start=final["date"], end=end_day, progress=False)
            if len(df):
                row = df.iloc[0]
                final["ohlc"] = {
                    "O": float(row["Open"]),
                    "H": float(row["High"]),
                    "L": float(row["Low"]),
                    "C": float(row["Close"]),
                    "V": float(row["Volume"]),
                }
                final["price_range"] = [float(row["Low"]), float(row["High"])]
    except Exception as e:
        print(f"yfinance enrichment skipped: {e}")
    with open("final_output.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    return final

# ─────────────────────────────────────────────────────────────
# Gradio pipeline
# ─────────────────────────────────────────────────────────────
def handle_request(query: str, image, do_news: bool = False):
    try:
        t0 = time.perf_counter()

        if image is not None:
            final_output = ensure_final_output_from_image(image)
        elif query and query.strip():
            final_output = ensure_final_output_from_text(query.strip())
        else:
            raise gr.Error("Please provide either a text query or upload a chart image.")

        ticker = final_output.get("ticker") or final_output.get("company_ticker")
        if not ticker:
            raise gr.Error("Could not determine ticker from JSON. Include a ticker in the text or ensure OCR extracted it.")

        anchor_date = final_output.get("date") or get_curday()
        exchange = (final_output.get("exchange") or "").strip()

        # Ensure OHLC exists so forecast has some context  ← NEW
        ensure_min_ohlc(final_output)

        # Sentiment from YOLO patterns
        sentiment = simple_sentiment_from_patterns(final_output.get("predictions", []))

        # Summary (LLM with sentence-style bullets, with deterministic fallback)
        try:
            summary_md = llm_brief_summary(final_output, sentiment)
        except Exception as e:
            print("LLM brief failed, using fallback:", e)
            summary_md = summarize_image(final_output, sentiment)

        # News (optional)
        BAD_TICKERS = {"N/A","NA","NONE","UNKNOWN","-",""}
        has_valid_ticker = bool(ticker) and ticker.strip().upper() not in BAD_TICKERS
        news_items = []
        if do_news and has_valid_ticker:
            news_items = news_for_window(ticker, anchor_date, exchange, weeks=1)

        # Forecast (ALWAYS run if we have a ticker; patterns are optional)  ← CHANGED
        prompt = build_llm_prompt(final_output, news_items, require_patterns=False)  # ← CHANGED
        forecast_md = llm_forecast_strong(final_output, news_items)

        # Save JSON for Download
        json_path = os.path.abspath("final_output.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        final_json_str = json.dumps(final_output, indent=2, ensure_ascii=False)

        print(f"[TIMER] Total handled in {time.perf_counter() - t0:.2f}s")

        return summary_md, sentiment, forecast_md, final_json_str, pd.DataFrame(news_items) if news_items else pd.DataFrame(
            [{"info": "News omitted or none found for the selected window."}]
        ), json_path

    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Failed to process request: {e}")



def chat_handle_turn_pairs(history_pairs, user_text: str, user_files: Optional[List[str]]):
    """
    Orchestrates a single chat turn.
    - Appends the user's message to the (user, assistant) tuple list
    - Calls handle_request() with either the uploaded image or the text
    - Appends an assistant markdown reply that merges summary + forecast
    - Returns: (updated_chat, summary_md, sentiment, forecast_md, final_json_str, news_df, json_download_path)
    """
    history_pairs = history_pairs or []

    # Pick first valid image path if any
    image_path = None
    if user_files:
        for f in user_files:
            if isinstance(f, str) and os.path.exists(f):
                image_path = f
                break

    # Construct a readable user bubble
    user_msg = (user_text or "").strip()
    if not user_msg and image_path:
        user_msg = "🖼️ Uploaded a stock chart image"
    if not user_msg:
        user_msg = "(no input provided)"

    # Tentatively add user turn
    history_pairs = history_pairs + [(user_msg, None)]

    # Run the core pipeline
    summary_md, sentiment, forecast_md, final_json_str, news_df, json_path = handle_request(user_text, image_path)

    # Compose assistant bubble (what the Chatbot shows)
    assistant_md = f"{summary_md}\n\n---\n\n### Forecast & Analysis\n{forecast_md}"

    # Fill in the assistant side of the last tuple
    history_pairs[-1] = (history_pairs[-1][0], assistant_md)

    # Gradio expects: chat, summary, sentiment, forecast, json, news_df, download_path
    return history_pairs, summary_md, sentiment, forecast_md, final_json_str, news_df, json_path


# Toggle handler for JSON accordion
def _toggle(prev_open: bool):
    new_open = not prev_open
    new_label = "Hide JSON Preview" if new_open else "Show JSON Preview"
    return gr.update(open=new_open), new_open, gr.update(value=new_label)

# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
with gr.Blocks(title="FinGPT-M — Chat") as demo:
    gr.HTML(
        """
        <style>
          .compact-container {max-width: 840px; margin: 0 auto;}
          .gradio-container {background: #0b0f17;}
          .gr-chatbot {border-radius: 14px;}
          .message-wrap .message {border-radius: 16px; padding: 10px 14px;}
          .btn-row {display:flex; gap:12px; flex-wrap:wrap;}
          .shadow-card {background: #0f1623; border: 1px solid #1f2a3a; border-radius: 16px; padding: 12px;}
        </style>
        """
    )

    with gr.Row():
        with gr.Column(elem_classes=["compact-container"], scale=8):
            gr.Markdown("## FinGPT-M\nChat with text or drop a **stock chart image**. I’ll extract metadata, run YOLO patterns, pull news, and generate a forecast.")

            # ✅ Simple, stable tuple-based Chatbot
            chat = gr.Chatbot(
                label=None,
                height=520,
                show_copy_button=True,
                bubble_full_width=False,
            )

            # Prefer MultimodalTextbox; fallback to Textbox+Image
            try:
                mm = gr.MultimodalTextbox(
                    placeholder="Ask: “Analyze TSLA (2025-06-04)” or drop a chart image…",
                    show_label=False,
                    file_types=["image"],
                    autofocus=True,
                    submit_btn=True,
                )
                _uses_mm = True
            except Exception:
                with gr.Row():
                    txt = gr.Textbox(placeholder="Type here…", scale=4)
                    img = gr.Image(type="filepath", label="", scale=2)
                    send = gr.Button("Send", variant="primary")
                _uses_mm = False

            # Hidden plumbing outputs
            summary_out = gr.Markdown(visible=False)
            sentiment_out = gr.Textbox(visible=False)
            forecast_out = gr.Markdown(visible=False)
            json_out = gr.Code(language="json", label="", visible=False)
            news_out = gr.Dataframe(visible=False)
            json_download = gr.DownloadButton(label="Download final_output.json", visible=False)

        # Right-side tools panel
        with gr.Column(min_width=340, scale=4):
            with gr.Group():
                gr.Markdown("### Tools")
                with gr.Column(elem_classes=["shadow-card"]):
                    gr.Markdown("**final_output.json**")
                    with gr.Accordion("Preview", open=False):
                        json_preview = gr.Code(language="json", label="", interactive=False)
                    drow = gr.Row(elem_classes=["btn-row"])
                    with drow:
                        json_download_side = gr.DownloadButton("Download JSON")
                with gr.Column(elem_classes=["shadow-card"]):
                    gr.Markdown("**Recent News**")
                    news_table = gr.Dataframe(wrap=True)

    # State: list of (user, assistant) tuples
    chat_state = gr.State([])

    # Wiring
    if _uses_mm:
        def on_mm_submit_pairs(history_pairs, data):
            user_text = (data or {}).get("text") or ""
            user_files = (data or {}).get("files") or []
            return chat_handle_turn_pairs(history_pairs, user_text, user_files)

        mm.submit(
            fn=on_mm_submit_pairs,
            inputs=[chat_state, mm],
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
    else:
        def on_send_click_pairs(history_pairs, text, image):
            return chat_handle_turn_pairs(history_pairs, text, [image] if image else None)

        send.click(
            fn=on_send_click_pairs,
            inputs=[chat_state, txt, img],
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

if __name__ == "__main__":
    if not os.path.exists(YOLO_WEIGHTS):
        raise FileNotFoundError(f"YOLO weights not found at: {YOLO_WEIGHTS}")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    demo.launch(share=True, debug=True)