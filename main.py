import os
import re
import json
import asyncio
from google import genai
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from calculator import compare_transfers, get_live_usd_to_bdt

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="RemitWise AI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str

# Notice we added 'data' here so the frontend can build the UI
class ChatResponse(BaseModel):
    response: str
    data: Optional[dict] = None 
    action: Optional[dict] = None

@app.get("/")
def home():
    return {"status": "Agent is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    message_lower = request.message.lower()

    # 1. Handle Troubleshooting
    if "delay" in message_lower or "not received" in message_lower:
        return ChatResponse(response="I can help with that. What is your transaction ID? (e.g., TRX-123)")

    # 2. Handle Rate Alerts
    if "alert" in message_lower or "notify" in message_lower:
        match_rate = re.search(r"(\d{2,3}\.\d+|\d{3})", message_lower)
        rate_val = float(match_rate.group(1)) if match_rate else 120.0
        return ChatResponse(
            response=f"I can set a live alert for you. Click below to get notified when the rate hits {rate_val}.",
            action={"type": "set_alert", "label": f"🔔 Set Alert for {rate_val}", "target_rate": rate_val}
        )

    # 3. Handle Transfer Calculations & General Chat
    match_amount = re.search(r"\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)", message_lower)
    calc_data = None

    if match_amount:
        amount = int(float(match_amount.group(1).replace(",", "")))
        calc_data = compare_transfers(amount)
        
        prompt = f"""
You are a helpful remittance assistant for Bangladesh.
User message: "{request.message}"

Here is the exact math and data calculated by our Python engine:
{json.dumps(calc_data, indent=2)}

CRITICAL INSTRUCTIONS:
1. Adapt to the user's language (English, Bangla, or Banglish).
2. Explain WHY the recommended provider is best using the data.
3. You MUST mention the "family_impact" (how much more money the family gets).
4. Mention the delivery speed of the recommended provider (e.g., "Wise takes 1-2 days").
5. Do not invent any numbers. Only use the numbers provided above.
"""
    else:
        # General conversational AI
        prompt = f"""
You are RemitWise AI, a helpful and polite financial assistant for Bangladesh.
You ONLY answer questions related to money transfers, remittances, exchange rates, and financial decisions. 
If the user asks about anything else, politely redirect them back to remittance advice.
User message: "{request.message}"
"""

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(client.models.generate_content, model='gemini-3.5-flash', contents=prompt),
            timeout=8.0
        )
        ai_text = response.text
    except Exception:
        # Graceful fallback if AI is slow
        if calc_data:
            ai_text = f"Math Results: {calc_data['recommended']} is best. {calc_data['family_impact']}"
        else:
            ai_text = "The AI is currently busy, but our math engine is still running."

    # Return both the AI's text and the structured data
    return ChatResponse(response=ai_text, data=calc_data, action=None)