import os
import re
import time
import json
import random
import finnhub
import torch
import gradio as gr
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from pynvml import *
from peft import PeftModel
from collections import defaultdict
from datetime import date, datetime, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import traceback

load_dotenv(override=True)
access_token = os.getenv("HF_TOKEN")
finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))
print(finnhub_client.api_key)
# access_token = os.environ["HF_TOKEN"]
# finnhub_client = finnhub.Client(api_key=os.environ["FINNHUB_API_KEY"])

# ===== Load model =====
base_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-chat-hf',
    token=access_token,
    cache_dir="/content/llama_cache",
    trust_remote_code=True,
    device_map="cpu",
    torch_dtype=torch.float16,
    offload_folder="offload/"
)
model = PeftModel.from_pretrained(
    base_model,
    'FinGPT/fingpt-forecaster_dow30_llama2-7b_lora',
    offload_folder="/content/offload",
    cache_dir="/content/llama_cache"
)
model = model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-2-7b-chat-hf',
    token=access_token
)
streamer = TextStreamer(tokenizer)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
SYSTEM_PROMPT = (
    "You are a seasoned stock market analyst. Your task is to list the positive developments and "
    "potential concerns for companies based on relevant news and basic financials from the past weeks, "
    "then provide an analysis and prediction for the companies' stock price movement for the upcoming week."
)

# ===== GPU monitoring =====
def print_gpu_utilization():
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")
    except:
        pass

def get_curday():
    return date.today().strftime("%Y-%m-%d")

def n_weeks_before(date_string, n):
    dt = datetime.strptime(date_string, "%Y-%m-%d") - timedelta(days=7*n)
    return dt.strftime("%Y-%m-%d")

# ===== Data fetching =====
def get_stock_data(stock_symbol, steps):
    stock_data = yf.download(stock_symbol, steps[0], steps[-1])
    if len(stock_data) == 0:
        raise gr.Error(f"Failed to download stock data for {stock_symbol}")
    dates, prices = [], []
    available_dates = stock_data.index.format()
    for date_str in steps[:-1]:
        for i in range(len(stock_data)):
            if available_dates[i] >= date_str:
                prices.append(stock_data['Close'].iloc[i])
                dates.append(datetime.strptime(available_dates[i], "%Y-%m-%d"))
                break
    dates.append(datetime.strptime(available_dates[-1], "%Y-%m-%d"))
    prices.append(stock_data['Close'].iloc[-1])
    return pd.DataFrame({
        "Start Date": dates[:-1], "End Date": dates[1:],
        "Start Price": prices[:-1], "End Price": prices[1:]
    })

def get_news(symbol, data):
    news_list = []
    for _, row in data.iterrows():
        start_date = row['Start Date'].strftime('%Y-%m-%d')
        end_date = row['End Date'].strftime('%Y-%m-%d')
        time.sleep(1)  # Finnhub rate limit
        weekly_news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
        weekly_news = [
            {
                "date": datetime.fromtimestamp(n['datetime']).strftime('%Y%m%d%H%M%S'),
                "headline": n['headline'],
                "summary": n['summary']
            } for n in weekly_news
        ]
        news_list.append(json.dumps(weekly_news))
    data['News'] = news_list
    return data

def get_company_prompt(symbol):
    profile = finnhub_client.company_profile2(symbol=symbol)
    company_template = (
        "[Company Introduction]:\n\n{name} is a leading entity in the {finnhubIndustry} sector. "
        "Incorporated and publicly traded since {ipo}, the company has established its reputation "
        "as one of the key players in the market. As of today, {name} has a market capitalization of "
        "{marketCapitalization:.2f} in {currency}, with {shareOutstanding:.2f} shares outstanding.\n\n"
        "{name} operates primarily in the {country}, trading under the ticker {ticker} on the {exchange}. "
        "As a dominant force in the {finnhubIndustry} space, the company continues to innovate and drive progress."
    )
    return company_template.format(**profile)

# ===== Prompt construction =====
def construct_prompt(ticker, curday, n_weeks, use_basics):
    steps = [n_weeks_before(curday, n) for n in range(n_weeks + 1)][::-1]
    data = get_stock_data(ticker, steps)
    data = get_news(ticker, data)
    company_prompt = get_company_prompt(ticker)
    prompt = B_INST + B_SYS + SYSTEM_PROMPT + E_SYS + company_prompt + E_INST
    return company_prompt, prompt

# ===== Model prediction =====
def predict(ticker, date_val, n_weeks, use_basics, chart_metadata=None):
    print_gpu_utilization()
    company_info, prompt = construct_prompt(ticker, date_val, n_weeks, use_basics)
    inputs = tokenizer(prompt, return_tensors='pt', padding=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    res = model.generate(
        **inputs, max_length=4000, do_sample=True,
        eos_token_id=tokenizer.eos_token_id, streamer=streamer
    )
    output = tokenizer.decode(res[0], skip_special_tokens=True)
    answer = re.sub(r'.*\[/INST\]\s*', '', output, flags=re.DOTALL)
    torch.cuda.empty_cache()

    chart_section = ""
    if chart_metadata:
        price_min = chart_metadata.get("price_range", {}).get("min")
        price_max = chart_metadata.get("price_range", {}).get("max")
        vol = chart_metadata.get("ohlc", {}).get("V")
        if price_min and price_max:
            chart_section += f"\n📊 Price Range: ${price_min} – ${price_max}"
        if vol:
            chart_section += f"\n📊 Volume: ~{vol:,} shares traded"

    final_output = (
        f"🏢 **Company Overview:**\n{company_info}\n\n"
        f"{answer.strip()}\n"
        f"{chart_section}\n\n"
        f"📄 _Source: FinGPT forecast integrating news, financials, and chart metadata._"
    )
    return final_output

# ===== Predict from JSON (image mode) =====
def predict_from_json(json_path, n_weeks=1, use_basics=True):
    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    ticker = metadata.get("ticker")
    extracted_date = datetime.today().strftime("%Y-%m-%d")
    chart_metadata = {
        "price_range": {"min": metadata.get("price_range", [None, None])[0],
                        "max": metadata.get("price_range", [None, None])[1]},
        "ohlc": metadata.get("ohlc", {})
    }
    return predict(ticker, extracted_date, n_weeks, use_basics, chart_metadata)

# def predict_from_json(json_path, use_basics=True):
#     with open(json_path, "r", encoding="utf-8") as f:
#         metadata = json.load(f)

#     ticker = metadata.get("ticker")
#     sessions = metadata.get("sessions", [])

#     # Pick first valid YYYY-MM-DD date from sessions
#     extracted_date = None
#     for s in sessions:
#         if "date" in s and re.match(r"\d{4}-\d{2}-\d{2}", s["date"]):
#             extracted_date = s["date"]
#             break
#     if not extracted_date:
#         extracted_date = get_curday()

#     # Chart metadata from JSON
#     chart_metadata = {
#         "price_range": {"min": metadata.get("price_range", [None, None])[0],
#                         "max": metadata.get("price_range", [None, None])[1]},
#         "ohlc": metadata.get("ohlc", {})
#     }

#     # Optional: company info from Finnhub
#     try:
#         company_info = get_company_prompt(ticker)
#     except:
#         company_info = f"[Company Introduction]:\n\nNo profile data available for {ticker}"

#     # Get stock data from yfinance for extracted date (±3 days window)
#     start_dt = (datetime.strptime(extracted_date, "%Y-%m-%d") - timedelta(days=3)).strftime("%Y-%m-%d")
#     end_dt = (datetime.strptime(extracted_date, "%Y-%m-%d") + timedelta(days=3)).strftime("%Y-%m-%d")

#     try:
#         stock_data = yf.download(ticker, start=start_dt, end=end_dt)
#         if not stock_data.empty:
#             stock_summary = f"\nStock close prices around {extracted_date}:\n" + \
#                             "\n".join([f"{idx.date()}: {row['Close']:.2f}"
#                                       for idx, row in stock_data.iterrows()])
#         else:
#             stock_summary = "\nNo stock data available."
#     except:
#         stock_summary = "\nFailed to fetch stock data."

#     # Run model prediction (pass extracted date instead of n_weeks logic)
#     forecast = predict(ticker, extracted_date, n_weeks=1, use_basics=use_basics, chart_metadata=chart_metadata)

#     # Final combined output
#     return (
#         f"🏢 **Company Overview:**\n{company_info}\n\n"
#         f"📅 **Date Used for Analysis:** {extracted_date}\n"
#         f"{stock_summary}\n\n"
#         f"{forecast}"
#     )

# pip install easyocr

# pip install ultralytics

# pip install holidays

# ===== Run OCR + YOLO for image =====
sys.path.append('/content/FinGPT-M/fingpt/stock_chart_trends_analysis')
from StockChart_Trend_Prediction import StockChartMetadataExtractor, StockChartTrendPredictor, combine_metadata_and_predictions

def run_chart_analysis(image_path):
    model_path = r"/content/FinGPT-M/fingpt/stock_chart_trends_analysis/best.pt"
    metadata_extractor = StockChartMetadataExtractor(image_path)
    metadata_extractor.save_metadata_to_json("metadata.json")
    stock_chart_predictor = StockChartTrendPredictor(model_path)
    output, _ = stock_chart_predictor.predict(image_path)
    stock_chart_predictor.save_predictions_to_json(output, 'predictions.json')
    combine_metadata_and_predictions('metadata.json', output, 'final_output.json')
    for file in ['metadata.json', 'predictions.json']:
        if os.path.exists(file):
            os.remove(file)
    return "final_output.json"

# ===== Gradio chatbot =====
with gr.Blocks() as demo:
    gr.Markdown("## 📊 FinGPT-Forecaster Chatbot")
    chatbot = gr.Chatbot(label="FinGPT", height=500)
    with gr.Row():
        query_box = gr.Textbox(
            placeholder="Ask: 'What's the prediction for TSLA after 2025-07-25 for past 2 weeks using financials?'",
            show_label=False, scale=5
        )
        image_upload = gr.Image(type="pil", label=None, scale=1)

    # def chat_handler(user_input, chat_history, image=None):
    #     chat_history.append((user_input, None))
    #     try:
    #         if image:
    #             image_path = "uploaded_chart.png"
    #             image.save(image_path)
    #             json_path = run_chart_analysis(image_path)
    #             formatted_output = predict_from_json(json_path, n_weeks=3, use_basics=True)
    #         else:
    #             words = re.findall(r'\b[A-Z.]{2,6}\b', user_input.upper())
    #             ticker = next((w for w in words if w.isupper()), "AAPL")
    #             date_match = re.search(r"\d{4}-\d{2}-\d{2}", user_input)
    #             date_val = date_match.group(0) if date_match else get_curday()
    #             n_weeks_match = re.search(r"past (\d+) week", user_input.lower())
    #             n_weeks = int(n_weeks_match.group(1)) if n_weeks_match else 3
    #             use_basics = bool(re.search(r"(financial|basic)", user_input.lower()))
    #             formatted_output = predict(ticker, date_val, n_weeks, use_basics)
    #         chat_history[-1] = (user_input, formatted_output)
    #     except Exception as e:
    #         chat_history[-1] = (user_input, f"❌ Error: {str(e)}")
    #     return "", chat_history, None
    def chat_handler(user_input, chat_history, image=None):
      chat_history.append((user_input, None))

      try:
          if image:
              # Force image mode: ignore textbox ticker/date
              image_path = "uploaded_chart.png"
              image.save(image_path)

              # Run OCR + YOLO → JSON
              json_path = run_chart_analysis(image_path)

              # Run JSON-based prediction
              formatted_output = predict_from_json(json_path, use_basics=True)

              # Replace chat message
              # chat_history[-1] = (f"[Image: {os.path.basename(image_path)}]", formatted_output)
              original_filename = getattr(image, 'name', 'Uploaded Image')
              chat_history[-1] = (f"[Image: {os.path.basename(original_filename)}]", formatted_output)
              # chat_history[-1] = (image_path, formatted_output)


              # Return with cleared textbox
              return "", chat_history, None

          else:
              # Text mode only
              words = re.findall(r'\b[A-Z.]{2,6}\b', user_input.upper())
              ticker = next((w for w in words if w.isupper()), "AAPL")
              date_match = re.search(r"\d{4}-\d{2}-\d{2}", user_input)
              date_val = date_match.group(0) if date_match else get_curday()
              n_weeks_match = re.search(r"past (\d+) week", user_input.lower())
              n_weeks = int(n_weeks_match.group(1)) if n_weeks_match else 3
              use_basics = bool(re.search(r"(financial|basic)", user_input.lower()))

              formatted_output = predict(ticker, date_val, n_weeks, use_basics)

              chat_history[-1] = (user_input, formatted_output)
              return "", chat_history, None

      except Exception as e:
          chat_history[-1] = (user_input, f"❌ Error: {str(e)}")
          return "", chat_history, None


    query_box.submit(fn=chat_handler,
                     inputs=[query_box, chatbot, image_upload],
                     outputs=[query_box, chatbot, image_upload])
    image_upload.change(fn=chat_handler,
                        inputs=[query_box, chatbot, image_upload],
                        outputs=[query_box, chatbot, image_upload])

demo.launch(share=True, debug=True)

