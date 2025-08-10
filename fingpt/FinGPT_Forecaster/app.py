import os
import re
import time
import json
import random
import traceback
from typing import Dict, Any, Optional, Tuple, List

import finnhub
import torch
import gradio as gr
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from datetime import date, datetime, timedelta
from collections import defaultdict

# ===== Optional forecaster (your LoRA on LLaMA-2) =====
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel

# ===== Your image -> JSON pipeline =====
# Ensure this file is importable (same folder or PYTHONPATH)
# and exposes StockChartTrendPredictor and StockChartMetadataExtractor
from StockChart_Trend_Prediction import StockChartTrendPredictor, StockChartMetadataExtractor

# ---------------------------
# ENV & CLIENTS
# ---------------------------
load_dotenv(override=True)
access_token = os.getenv("HF_TOKEN")
finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))

# ---------------------------
# MODEL (optional; guarded)
# ---------------------------
USE_GENERATIVE_FORECASTER = True  # flip to False if you want to skip the LLM step

model, tokenizer, streamer = None, None, None
if USE_GENERATIVE_FORECASTER:
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            token=access_token,
            cache_dir="E:/FinGPT/llama_cache",
            trust_remote_code=True,
            device_map="cpu",               # keep CPU-safe; move to "auto" if you have GPU
            torch_dtype=torch.float16,
            offload_folder="offload/"
        )
        model = PeftModel.from_pretrained(
            base_model,
            "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora",
            offload_folder="E:/FinGPT/offload/",
            cache_dir="E:/FinGPT/llama_cache"
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            token=access_token
        )
        streamer = TextStreamer(tokenizer)
    except Exception as e:
        print("⚠️ Could not initialize LLM forecaster. Falling back to rule-based summary only.\n", e)
        USE_GENERATIVE_FORECASTER = False

# ---------------------------
# PROMPTS
# ---------------------------
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SYSTEM_PROMPT = (
    "You are a seasoned stock market analyst. Use only the structured JSON facts provided "
    "(ticker, time ranges, OCR metadata, OHLC ranges, detected chart patterns) and recent company news. "
    "List 2–4 positive developments and 2–4 potential concerns, then give a short forecast-style analysis "
    "for the upcoming sessions. Keep the response concise and factual."
)

# ---------------------------
# HELPERS
# ---------------------------
def get_curday() -> str:
    return date.today().strftime("%Y-%m-%d")

def parse_query(q: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract ticker & date from free text like:
    'what is the stock analysis for TSLA on 2025-06-04'
    Returns (ticker, date) where date is YYYY-MM-DD or None.
    """
    if not q:
        return None, None
    # naive ticker guess: last 2–5 contiguous caps letters/numbers
    tickers = re.findall(r"\b[A-Z]{1,5}\b", q.upper())
    ticker = tickers[-1] if tickers else None

    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", q)
    day = m.group(1) if m else None
    return ticker, day

def n_weeks_before(date_string: str, n: int) -> str:
    dt = datetime.strptime(date_string, "%Y-%m-%d") - timedelta(days=7 * n)
    return dt.strftime("%Y-%m-%d")

def get_stock_data(stock_symbol: str, steps: List[str]) -> pd.DataFrame:
    stock_data = yf.download(stock_symbol, steps[0], steps[-1])
    if len(stock_data) == 0:
        raise gr.Error(f"Failed to download stock price data for symbol {stock_symbol} from yfinance!")
    dates, prices = [], []
    available_dates = stock_data.index.format()
    for d in steps[:-1]:
        for i in range(len(stock_data)):
            if available_dates[i] >= d:
                prices.append(stock_data["Close"].iloc[i])
                dates.append(datetime.strptime(available_dates[i], "%Y-%m-%d"))
                break
    dates.append(datetime.strptime(available_dates[-1], "%Y-%m-%d"))
    prices.append(stock_data["Close"].iloc[-1])
    return pd.DataFrame(
        {"Start Date": dates[:-1], "End Date": dates[1:], "Start Price": prices[:-1], "End Price": prices[1:]}
    )

def get_company_news(symbol: str, start_date: str, end_date: str) -> List[Dict[str, str]]:
    # Finnhub date format: YYYY-MM-DD
    weekly_news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
    # Convert/clean
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
    """
    Pull news for [anchor_day - weeks, anchor_day].
    """
    try:
        end_dt = datetime.strptime(anchor_day, "%Y-%m-%d")
    except Exception:
        end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=7 * weeks)
    return get_company_news(symbol, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))

def simple_sentiment_from_patterns(preds: List[Dict[str, Any]]) -> str:
    """
    Heuristic sentiment from detected chart patterns by class name.
    Map your YOLO class names to bullish/bearish/neutral.
    """
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
    if score > 0:
        return "Positive"
    if score < 0:
        return "Negative"
    return "Neutral"

def llm_forecast_from_json(final_output: Dict[str, Any], ticker: Optional[str], news_snippets: List[Dict[str, str]]) -> str:
    """
    Use your LoRA forecaster to produce a polished forecast/summary from final_output JSON.
    Falls back to rule-based text if LLM is disabled/unavailable.
    """
    # Build a compact context from JSON
    ctx = {
        "ticker": ticker or final_output.get("ticker") or final_output.get("company_ticker"),
        "exchange": final_output.get("exchange"),
        "ohlc": final_output.get("ohlc"),
        "sessions": final_output.get("sessions"),
        "price_range": final_output.get("price_range"),
        "predictions": final_output.get("predictions"),
    }

    news_text = "\n".join(
        f"- {n['date']} | {n['headline']} :: {n['summary'][:220]}..." for n in (news_snippets or [])[:6]
    )
    user_prompt = (
        "Here is structured JSON from a stock chart OCR + trend detector, followed by recent company news.\n\n"
        f"[JSON]\n{json.dumps(ctx, indent=2)}\n\n"
        f"[NEWS]\n{news_text if news_text else 'No recent company-specific news found in window.'}\n\n"
        "Provide: (1) Positive Developments (2–4 bullets), (2) Potential Concerns (2–4 bullets), "
        "(3) A brief forecast-style analysis for upcoming sessions."
    )

    if not USE_GENERATIVE_FORECASTER or model is None or tokenizer is None:
        # Fallback: rule-based concise synthesis
        preds = final_output.get("predictions", [])
        sent = simple_sentiment_from_patterns(preds)
        bullet_pos = []
        bullet_neg = []

        # naive bullets from predictions
        for p in preds[:3]:
            nm = p.get("class", "pattern")
            conf = p.get("confidence", 0.0)
            if sent == "Positive":
                bullet_pos.append(f"Detected {nm} (conf {conf:.2f}) supports near-term upside.")
            elif sent == "Negative":
                bullet_neg.append(f"Detected {nm} (conf {conf:.2f}) flags potential downside.")
            else:
                bullet_pos.append(f"Pattern {nm} (conf {conf:.2f}) observed; directional significance uncertain.")

        if news_snippets:
            headline = news_snippets[0]["headline"]
            bullet_pos.append(f"Recent news: {headline}")

        analysis = (
            "Overall stance: "
            f"{'Cautiously bullish' if sent=='Positive' else 'Cautiously bearish' if sent=='Negative' else 'Neutral/sideways'} "
            "based on detected patterns and recent context."
        )
        return (
            "### Positive Developments\n- " + "\n- ".join(bullet_pos or ["No clear upside catalysts."]) +
            "\n\n### Potential Concerns\n- " + "\n- ".join(bullet_neg or ["No prominent bearish signals."]) +
            f"\n\n### Forecast & Analysis\n{analysis}"
        )

    # LLM path
    prompt = B_INST + B_SYS + SYSTEM_PROMPT + E_SYS + user_prompt + E_INST
    inputs = tokenizer(prompt, return_tensors="pt", padding=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    gen = model.generate(
        **inputs, max_length=1800, do_sample=True, eos_token_id=tokenizer.eos_token_id, use_cache=True, streamer=None
    )
    out = tokenizer.decode(gen[0], skip_special_tokens=True)
    # strip anything before closing [/INST]
    out = re.sub(r".*\[/INST\]\s*", "", out, flags=re.DOTALL)
    return out.strip()

def ensure_final_output_from_image(image_path: str) -> Dict[str, Any]:
    """
    Runs your chart OCR + YOLO predictor and returns a final_output dict.
    """
    metadata_extractor = StockChartMetadataExtractor(image_path)
    metadata = metadata_extractor.extract_metadata()  # should include ticker/company/exchange/sessions/price_range/ohlc...
    # You may want to resolve ticker if only company_name is present
    predictor = StockChartTrendPredictor(os.getenv("YOLO_MODEL_PATH", "best.pt"))
    preds, img = predictor.predict(image_path)
    # Save into JSON file (as your code does)—but also return the merged dict directly
    final = predictor.save_predictions_to_json(preds, "final_output.json", img, metadata)
    return final

def ensure_final_output_from_text(query: str) -> Dict[str, Any]:
    """
    For text-only flows, produce a minimal final_output skeleton
    so downstream steps are uniform.
    """
    ticker, day = parse_query(query)
    final = {
        "ticker": ticker,
        "date": day or get_curday(),
        "source": "text_query",
        "predictions": [],              # none from chart; can be filled by other modules later
        "sessions": [],
        "price_range": [None, None],
        "ohlc": {},
        "exchange": None,
        "company_name": None,
    }
    # Optional: attempt to enrich OHLC for the day
    try:
        if ticker:
            df = yf.download(ticker, start=final["date"], end=(datetime.strptime(final["date"], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"))
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
    except Exception:
        pass

    # Write to disk for interoperability with your other tools
    with open("final_output.json", "w") as f:
        json.dump(final, f, indent=2)
    return final

def summarize_final_output(final_output: Dict[str, Any]) -> str:
    """
    Human-readable summary block from final_output JSON.
    """
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

# ---------------------------
# GRADIO PIPELINE
# ---------------------------
def handle_request(query: str, image) -> Tuple[str, str, str, str, pd.DataFrame]:
    """
    Returns:
      - summary_md
      - sentiment_str
      - forecast_md
      - final_output_json_str (pretty)
      - news_df
    """
    try:
        final_output: Dict[str, Any] = {}
        ticker: Optional[str] = None
        anchor_date: str = get_curday()

        if image is not None:
            # Image path from temp file
            final_output = ensure_final_output_from_image(image)
            ticker = final_output.get("ticker") or final_output.get("company_ticker")
            anchor_date = final_output.get("date") or anchor_date
        elif query and query.strip():
            final_output = ensure_final_output_from_text(query.strip())
            ticker = final_output.get("ticker")
            anchor_date = final_output.get("date") or anchor_date
        else:
            raise gr.Error("Please provide either a text query or upload a chart image.")

        # Sentiment from predictions (heuristic)
        preds = final_output.get("predictions", [])
        sentiment = simple_sentiment_from_patterns(preds)

        # News
        news_items = news_for_window(ticker, anchor_date, weeks=1) if ticker else []
        news_df = pd.DataFrame(news_items) if news_items else pd.DataFrame(
            [{"info": "No recent company-specific news found for the selected window."}]
        )

        # Forecaster-style writeup (LLM if available, else rule-based)
        forecast_md = llm_forecast_from_json(final_output, ticker, news_items)

        # Summary
        summary_md = summarize_final_output(final_output)

        # Pretty JSON
        final_json_str = json.dumps(final_output, indent=2, ensure_ascii=False)

        return summary_md, sentiment, forecast_md, final_json_str, news_df

    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Failed to process request: {e}")

# ---------------------------
# UI
# ---------------------------
with gr.Blocks(title="FinGPT-M: Text/Image → JSON → Summary + Sentiment + Forecast + News") as demo:
    gr.Markdown(
        "### FinGPT-M\n"
        "Provide **either** a free-text query (e.g., `What is the stock analysis for TSLA on 2025-06-04`) "
        "**or** upload a **stock chart image**. We’ll produce `final_output` JSON, summarize it, "
        "estimate sentiment, generate a forecaster-style analysis, and show recent news."
    )

    with gr.Row():
        query_in = gr.Textbox(
            label="Text Query (optional)",
            placeholder="e.g., What is the stock analysis for TSLA on 2025-06-04",
            lines=2,
        )
        image_in = gr.Image(label="Upload Stock Chart (optional)", type="filepath")

    submit = gr.Button("Analyze", variant="primary")

    with gr.Row():
        summary_out = gr.Markdown(label="Summary (from final_output)")
        sentiment_out = gr.Textbox(label="Sentiment (heuristic from patterns)", interactive=False)

    forecast_out = gr.Markdown(label="Forecaster-style Analysis")
    json_out = gr.Code(label="final_output.json (preview)", language="json")
    news_out = gr.Dataframe(label="News (Finnhub)", wrap=True)

    submit.click(
        fn=handle_request,
        inputs=[query_in, image_in],
        outputs=[summary_out, sentiment_out, forecast_out, json_out, news_out],
        api_name="analyze",
    )

if __name__ == "__main__":
    # YOLO model path for image flow; override with env var YOLO_MODEL_PATH if needed
    os.environ.setdefault("YOLO_MODEL_PATH", "E:/FinGPT-M/fingpt/stock_chart_trends_analysis/best.pt")
    demo.launch(share=False)
