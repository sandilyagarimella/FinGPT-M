# app.py — FinGPT-M UI (LLM Summary + Forecast, JSON download + toggle preview)

import os
import re
import json
import time
import traceback
from typing import Dict, Any, Optional, Tuple, List

import sys
# >>> adjust to your local repo layout
sys.path.insert(0, r"E:\FinGPT-M\fingpt\stock_chart_trends_analysis")

import torch
import gradio as gr
import pandas as pd
import yfinance as yf
import finnhub
from dotenv import load_dotenv
from datetime import date, datetime, timedelta

# ─────────────────────────────────────────────────────────────
# Paths & env
# ─────────────────────────────────────────────────────────────
YOLO_WEIGHTS = r"E:\FinGPT-M\fingpt\stock_chart_trends_analysis\best.pt"
os.environ["YOLO_MODEL_PATH"] = YOLO_WEIGHTS
os.environ["ULTRALYTICS_VERBOSE"] = "False"

load_dotenv(override=True)

HF_TOKEN = os.getenv("HF_TOKEN")
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
HF_ENDPOINT_URL = os.getenv("HF_ENDPOINT_URL")  # full endpoint URL

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set. Put it in your environment or .env")
if not FINNHUB_KEY:
    raise RuntimeError("FINNHUB_API_KEY not set. Put it in your environment or .env")
if not HF_ENDPOINT_URL:
    raise RuntimeError("HF_ENDPOINT_URL not set. Put your full Inference Endpoint URL into .env")

# ─────────────────────────────────────────────────────────────
# Imports from your pipeline
# ─────────────────────────────────────────────────────────────
from StockChart_Trend_Prediction import StockChartTrendPredictor, StockChartMetadataExtractor

# ─────────────────────────────────────────────────────────────
# Finnhub client (for news)
# ─────────────────────────────────────────────────────────────
finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)

# ─────────────────────────────────────────────────────────────
# LLM client (HF endpoint) + HTTP fallback
# ─────────────────────────────────────────────────────────────
from huggingface_hub import InferenceClient
import requests

print("• Initializing Hugging Face Inference Client…")
hf_client = InferenceClient(base_url=HF_ENDPOINT_URL, token=HF_TOKEN)
print("• Hugging Face Inference Client ready.")

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# Markers to fence model output (for both summary & forecast)
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

def get_company_news(symbol: str, start_date: str, end_date: str) -> List[Dict[str, str]]:
    weekly_news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
    items = [
        {
            "date": datetime.fromtimestamp(n["datetime"]).strftime("%Y-%m-%d %H:%M:%S"),
            "headline": n.get("headline", ""),
            "summary": n.get("summary", ""),
            "source": n.get("source", ""),
            "url": n.get("url", ""),
        }
        for n in (weekly_news or [])
        if not str(n.get("summary", "")).startswith("Looking for stock market analysis")
    ]
    return items

def news_for_window(symbol: str, anchor_day: str, weeks: int = 1) -> List[Dict[str, str]]:
    try:
        end_dt = datetime.strptime(anchor_day, "%Y-%m-%d")
    except Exception:
        end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=7 * weeks)
    return get_company_news(symbol, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))

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

def _auth_headers():
    return {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

def generate_text(prompt: str, max_new_tokens=160, temperature=0.2, top_p=0.95, stop=None, timeout=60) -> str:
    """Try TGI /generate first, then OpenAI-style /v1/completions."""
    stop = stop or []
    base = HF_ENDPOINT_URL.rstrip("/")

    # Try TGI
    try:
        tgi_url = f"{base}/generate"
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": temperature > 0,
                "repetition_penalty": 1.05,
                "return_full_text": False,
            },
        }
        if stop:
            payload["parameters"]["stop"] = stop
        r = requests.post(tgi_url, headers=_auth_headers(), json=payload, timeout=timeout)
        if r.ok:
            data = r.json()
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"] or ""
            if isinstance(data, list) and len(data) and "generated_text" in data[0]:
                return data[0]["generated_text"] or ""
            if isinstance(data, dict) and "text" in data:
                return data["text"] or ""
        else:
            print(f"[TGI] HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[TGI error] {e}")

    # Try OpenAI-style
    try:
        oa_url = f"{base}/v1/completions"
        payload = {
            "model": "local",
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop or None,
        }
        r = requests.post(oa_url, headers=_auth_headers(), json=payload, timeout=timeout)
        if r.ok:
            data = r.json()
            return (data.get("choices") or [{}])[0].get("text") or ""
        else:
            print(f"[OpenAI] HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[OpenAI error] {e}")

    return ""

# Fallback deterministic summary
def summarize_image(final_output: Dict[str, Any], sentiment: str) -> str:
    tkr = final_output.get("ticker") or final_output.get("company_ticker") or "—"
    name = (final_output.get("company_name") or "").rstrip(":") or "Unknown company"
    exch = final_output.get("exchange") or "—"
    pr = final_output.get("price_range") or [None, None]
    ohlc = final_output.get("ohlc") or {}
    preds = final_output.get("predictions", [])
    patterns = ", ".join([p.get("class","") for p in preds]) if preds else "None"
    o = ohlc.get("O", "—"); h = ohlc.get("H", "—"); l = ohlc.get("L", "—"); c = ohlc.get("C", "—"); v = ohlc.get("V", "—")

    return (
        f"### Image Summary\n"
        f"**{name}** ({tkr}) — **{exch}**\n\n"
        f"- Chart price range: **{pr[0]} – {pr[1]}**\n"
        f"- OHLC: **O {o}, H {h}, L {l}, C {c}, V {v}**\n"
        f"- Patterns: **{patterns}** · Sentiment: **{sentiment}**"
    )

# FinGPT-written brief summary (safe & bounded)
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
        "You are a precise market assistant. Using ONLY the JSON, write 4–6 concise markdown bullets "
        "that add interpretation (e.g., tight/wide range %, presence/absence of patterns). "
        "No HTML/code/extra sections. Keep total under 80 words."
    )

    # Attempt 1 (with fences)
    user_msg_1 = (
        f"[JSON]\n{json.dumps(ctx, indent=2)}\n\n"
        f"{OUTPUT_BEGIN}\n- **Label**: point\n- **Label**: point\n{OUTPUT_END}"
    )
    prompt_1 = f"{B_SYS}{sys_msg}{E_SYS}{B_INST}{user_msg_1}{E_INST}"
    raw1 = generate_text(prompt_1, max_new_tokens=180, temperature=0.2, top_p=0.95, stop=[OUTPUT_END])
    print("LLM Summary (raw1):", raw1[:300].replace("\n", " ⏎ "))

    summary = _between_markers(raw1).strip()

    # Attempt 2 (no fences) if empty
    if not summary:
        user_msg_2 = (
            f"[JSON]\n{json.dumps(ctx, indent=2)}\n\n"
            "Write 4–6 markdown bullets with **bold labels**. No preface, no headings."
        )
        prompt_2 = f"{B_SYS}{sys_msg}{E_SYS}{B_INST}{user_msg_2}{E_INST}"
        raw2 = generate_text(prompt_2, max_new_tokens=160, temperature=0.2, top_p=0.95)
        print("LLM Summary (raw2):", raw2[:300].replace("\n", " ⏎ "))
        lines = [ln for ln in raw2.splitlines() if ln.strip().startswith("- ")]
        summary = "\n".join(lines[:6]).strip()

    if not summary:
        return summarize_image(final_output, sentiment)

    return "### Image Summary\n" + summary

def build_llm_prompt(final_output, news_snippets):
    ctx = {
        "ticker": final_output.get("ticker"),
        "exchange": final_output.get("exchange"),
        "ohlc": final_output.get("ohlc"),
        "sessions": final_output.get("sessions"),
        "price_range": final_output.get("price_range"),
        "predictions": final_output.get("predictions"),
    }
    news_text = "\n".join(
        f"- {n['date']} | {n['headline']} :: {n['summary'][:200]}..."
        for n in (news_snippets or [])[:6]
    )

    system = (
        "You are a seasoned stock market analyst. Use ONLY the JSON and NEWS below. "
        "Write exactly three markdown sections: Positive Developments, Potential Concerns, Forecast & Analysis. "
        "Do not add any other sections, links, or unrelated commentary."
    )

    user = (
        f"[JSON]\n{json.dumps(ctx, indent=2)}\n\n"
        f"[NEWS]\n{news_text if news_text else 'No recent news'}\n\n"
        f"{OUTPUT_BEGIN}\n<Your answer here>\n{OUTPUT_END}"
    )
    # IMPORTANT: Llama-2 chat format for your LoRA
    return f"{B_SYS}{system}{E_SYS}{B_INST}{user}{E_INST}"

def llm_forecast(prompt: str) -> str:
    raw = generate_text(prompt, max_new_tokens=500, temperature=0.7, top_p=0.9, stop=[OUTPUT_END])
    return _between_markers(raw)

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
        json.dump(final, f, indent=2)
    return final

# ─────────────────────────────────────────────────────────────
# Gradio pipeline
# ─────────────────────────────────────────────────────────────
def handle_request(query: str, image):
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

        # Sentiment from YOLO patterns
        sentiment = simple_sentiment_from_patterns(final_output.get("predictions", []))

        # Prefer FinGPT brief summary; fall back if anything goes wrong
        try:
            summary_md = llm_brief_summary(final_output, sentiment)
        except Exception as e:
            print("LLM brief failed, using fallback:", e)
            summary_md = summarize_image(final_output, sentiment)

        # News
        news_items = news_for_window(ticker, anchor_date, weeks=1)
        news_df = pd.DataFrame(news_items) if news_items else pd.DataFrame(
            [{"info": "No recent company-specific news found for the selected window."}]
        )

        # Forecast (JSON + News)
        prompt = build_llm_prompt(final_output, news_items)
        forecast_md = llm_forecast(prompt)

        # Ensure JSON is on disk for the Download button
        json_path = os.path.abspath("final_output.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        final_json_str = json.dumps(final_output, indent=2, ensure_ascii=False)

        t1 = time.perf_counter()
        print(f"[TIMER] Total handled in {t1 - t0:.2f}s")

        # Return 6 outputs: summary, sentiment, forecast, json preview text, news df, download path
        return summary_md, sentiment, forecast_md, final_json_str, news_df, json_path

    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Failed to process request: {e}")

# Toggle handler for JSON accordion
def _toggle(prev_open: bool):
    new_open = not prev_open
    new_label = "Hide JSON Preview" if new_open else "Show JSON Preview"
    return gr.update(open=new_open), new_open, gr.update(value=new_label)

# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
with gr.Blocks(title="FinGPT-M: Text/Image → JSON → LLM Forecast + News") as demo:
    gr.Markdown("### FinGPT-M (LLM + News)\nProvide **text** or upload a **stock chart image**.")

    with gr.Row():
        query_in = gr.Textbox(
            label="Text Query",
            placeholder="e.g., What is the stock analysis for TSLA on 2025-06-04",
            lines=2
        )
        image_in = gr.Image(label="Upload Stock Chart (optional)", type="filepath")

    submit = gr.Button("Analyze", variant="primary")

    with gr.Row():
        summary_out = gr.Markdown(label="Image Summary")
        sentiment_out = gr.Textbox(label="Sentiment (from patterns)", interactive=False)

    forecast_out = gr.Markdown(label="LLM Forecast (JSON + News)")

    # Buttons row
    with gr.Row():
        json_download = gr.DownloadButton(label="Download final_output.json")
        toggle_preview_btn = gr.Button("Show JSON Preview")
        json_open_state = gr.State(False)

    # Collapsible preview (hidden by default)
    with gr.Accordion("final_output.json (preview)", open=False) as json_accordion:
        json_out = gr.Code(label="", language="json", interactive=False)

    news_out = gr.Dataframe(label="News")

    # Wiring
    submit.click(
        fn=handle_request,
        inputs=[query_in, image_in],
        outputs=[summary_out, sentiment_out, forecast_out, json_out, news_out, json_download],
        api_name="analyze",
    )

    toggle_preview_btn.click(
        fn=_toggle,
        inputs=[json_open_state],
        outputs=[json_accordion, json_open_state, toggle_preview_btn],
    )

if __name__ == "__main__":
    if not os.path.exists(YOLO_WEIGHTS):
        raise FileNotFoundError(f"YOLO weights not found at: {YOLO_WEIGHTS}")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    demo.launch(share=True, debug=True)