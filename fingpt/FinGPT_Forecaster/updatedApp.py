# app.py (Colab T4-optimized: YOLO + LLM on GPU, News enabled)

import os
import re
import json
import time
import traceback
from typing import Dict, Any, Optional, Tuple, List

import sys
sys.path.insert(0, "/content/FinGPT-M/fingpt/stock_chart_trends_analysis")  # your module folder

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
YOLO_WEIGHTS = "/content/FinGPT-M/fingpt/stock_chart_trends_analysis/best.pt"
os.environ["YOLO_MODEL_PATH"] = YOLO_WEIGHTS
os.environ["ULTRALYTICS_VERBOSE"] = "False"

load_dotenv(override=True)

# Prefer Colab secrets if available; fall back to env
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN") or os.getenv("HF_TOKEN")
    FINNHUB_KEY = userdata.get("FINNHUB_API_KEY") or os.getenv("FINNHUB_API_KEY")
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN")
    FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set. Set it via Colab userdata or environment.")

if not FINNHUB_KEY:
    raise RuntimeError("FINNHUB_API_KEY not set. Set it via Colab userdata or environment.")

# ─────────────────────────────────────────────────────────────
# Imports from your pipeline
# ─────────────────────────────────────────────────────────────
from StockChart_Trend_Prediction import StockChartTrendPredictor, StockChartMetadataExtractor

# ─────────────────────────────────────────────────────────────
# Finnhub client (MANDATORY; we fetch news)
# ─────────────────────────────────────────────────────────────
finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)

# ─────────────────────────────────────────────────────────────
# LLM forecaster (Llama-2-7B chat + LoRA), 4-bit on GPU (fast)
# ─────────────────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available. Switch Colab runtime to GPU (T4).")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("• Initializing LLM forecaster (4-bit on GPU)…")
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    token=HF_TOKEN,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=bnb_config,
)
model = PeftModel.from_pretrained(
    base_model,
    "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora",
).eval()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=HF_TOKEN)
print("• LLM ready on", torch.cuda.get_device_name(0))

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def get_curday() -> str:
    return date.today().strftime("%Y-%m-%d")

def parse_query(q: str) -> Tuple[Optional[str], Optional[str]]:
    if not q:
        return None, None
    # try (TICKER)
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
        for n in weekly_news or []
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
    # Heuristic mapping if YOLO finds patterns
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

def summarize_final_output(final_output: Dict[str, Any]) -> str:
    tkr = final_output.get("ticker") or final_output.get("company_ticker") or "N/A"
    name = final_output.get("company_name") or "N/A"
    exch = final_output.get("exchange") or "N/A"
    pr = final_output.get("price_range") or [None, None]
    ohlc = final_output.get("ohlc") or {}
    preds = final_output.get("predictions", [])
    lines = [
        f"**Ticker**: {tkr}  |  **Company**: {name}  |  **Exchange**: {exch}",
        f"**Price range (chart)**: {pr[0]} – {pr[1]}",
        f"**OHLC (OCR/enriched)**: {ohlc if ohlc else 'N/A'}",
        f"**Detected patterns**: {', '.join([p.get('class','') for p in preds]) if preds else 'None'}",
    ]
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────
# Build final_output.json
# ─────────────────────────────────────────────────────────────
def ensure_final_output_from_image(image_path: str) -> Dict[str, Any]:
    if not os.path.exists(YOLO_WEIGHTS):
        raise gr.Error(f"YOLO weights not found at: {YOLO_WEIGHTS}")
    # 1) OCR metadata (make sure easyocr.Reader(..., gpu=True) inside your OCR script)
    print(f"Attempting to load image from: {image_path}") # Add print statement
    metadata_extractor = StockChartMetadataExtractor(image_path) # Use the image_path provided by Gradio
    metadata = metadata_extractor.extract_metadata()

    # 2) YOLO predictions on GPU
    predictor = StockChartTrendPredictor(YOLO_WEIGHTS)
    try:
        predictor.model.to("cuda")
    except Exception:
        pass
    preds, img = predictor.predict(image_path)

    # 3) Merge + save
    final = predictor.save_predictions_to_json(preds, "final_output.json", img, metadata)

    # 4) Fallback ticker from OCR title if missing (e.g., "Apple Inc: (AAPL)")
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
    # Light OHLC enrichment (fast, same-day)
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
    with open("final_output.json", "w") as f:
        json.dump(final, f, indent=2)
    return final

# ─────────────────────────────────────────────────────────────
# LLM forecast (uses JSON + News)
# ─────────────────────────────────────────────────────────────
def build_llm_prompt(final_output: Dict[str, Any], news_snippets: List[Dict[str, str]]) -> str:
    ctx = {
        "ticker": final_output.get("ticker") or final_output.get("company_ticker"),
        "exchange": final_output.get("exchange"),
        "ohlc": final_output.get("ohlc"),
        "sessions": final_output.get("sessions"),
        "price_range": final_output.get("price_range"),
        "predictions": final_output.get("predictions"),
    }
    news_text = "\n".join(
        f"- {n['date']} | {n['headline']} :: {n['summary'][:240]}..." for n in (news_snippets or [])[:6]
    )
    system = (
        "You are a seasoned stock market analyst. Use only the JSON facts and provided company news. "
        "First list 2–4 Positive Developments and 2–4 Potential Concerns, then provide a short Forecast & Analysis "
        "for the near term. Be concise and factual."
    )
    user = (
        f"[JSON]\n{json.dumps(ctx, indent=2)}\n\n"
        f"[NEWS]\n{news_text if news_text else 'No recent company-specific news found in window.'}\n\n"
        "Return the three sections with clear headings."
    )
    return f"<<SYS>>\n{system}\n<</SYS>>\n\n{user}"

def llm_forecast(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=300,        # fast but informative
        do_sample=True, top_p=0.9, temperature=0.7,
        eos_token_id=tokenizer.eos_token_id, use_cache=True
    )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)

# ─────────────────────────────────────────────────────────────
# Gradio pipeline
# ─────────────────────────────────────────────────────────────
def handle_request(query: str, image):
    try:
        t0 = time.perf_counter()

        # Build/obtain final_output
        if image is not None:
            # Use the temporary file path provided by Gradio
            final_output = ensure_final_output_from_image(image)
        elif query and query.strip():
            final_output = ensure_final_output_from_text(query.strip())
        else:
            raise gr.Error("Please provide either a text query or upload a chart image.")

        ticker = final_output.get("ticker") or final_output.get("company_ticker")
        if not ticker:
            raise gr.Error("Could not determine ticker from JSON. Ensure the chart OCR extracted a ticker or include one in the text query.")

        anchor_date = final_output.get("date") or get_curday()

        # Sentiment (from patterns)
        sentiment = simple_sentiment_from_patterns(final_output.get("predictions", []))

        # News (always fetch; your requirement)
        news_items = news_for_window(ticker, anchor_date, weeks=1)
        news_df = pd.DataFrame(news_items) if news_items else pd.DataFrame(
            [{"info": "No recent company-specific news found for the selected window."}]
        )

        # LLM forecast using JSON + News
        prompt = build_llm_prompt(final_output, news_items)
        forecast_md = llm_forecast(prompt)

        # Summary and JSON preview
        summary_md = summarize_final_output(final_output)
        final_json_str = json.dumps(final_output, indent=2, ensure_ascii=False)

        t1 = time.perf_counter()
        print(f"[TIMER] Total handled in {t1 - t0:.2f}s")

        return summary_md, sentiment, forecast_md, final_json_str, news_df

    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Failed to process request: {e}")

# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
with gr.Blocks(title="FinGPT-M: Text/Image → JSON → LLM Forecast + Sentiment + News") as demo:
    gr.Markdown("### FinGPT-M (LLM + News)\nProvide **text** or upload a **stock chart image**.")

    with gr.Row():
        query_in = gr.Textbox(label="Text Query (optional)", placeholder="e.g., What is the stock analysis for TSLA on 2025-06-04", lines=2)
        image_in = gr.Image(label="Upload Stock Chart (optional)", type="filepath")

    submit = gr.Button("Analyze", variant="primary")

    with gr.Row():
        summary_out = gr.Markdown(label="Summary (from final_output)")
        sentiment_out = gr.Textbox(label="Sentiment (from patterns)", interactive=False)

    forecast_out = gr.Markdown(label="LLM Forecast (JSON + News)")
    json_out = gr.Code(label="final_output.json (preview)", language="json")
    news_out = gr.Dataframe(label="News", wrap=True)

    submit.click(
        fn=handle_request,
        inputs=[query_in, image_in],
        outputs=[summary_out, sentiment_out, forecast_out, json_out, news_out],
        api_name="analyze",
    )

if __name__ == "__main__":
    if not os.path.exists(YOLO_WEIGHTS):
        raise FileNotFoundError(f"YOLO weights not found at: {YOLO_WEIGHTS}")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    demo.launch(share=True, debug=True)