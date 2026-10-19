import os
import re
import json
import asyncio
import joblib
from google import genai
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from calculator import compare_transfers, get_live_usd_to_bdt

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load the fraud model
fraud_model = joblib.load("fraud_model.pkl")

app = FastAPI(title="FundGuide AI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    data: Optional[dict] = None 
    action: Optional[dict] = None

class FraudRequest(BaseModel):
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

@app.get("/")
def home():
    return {"status": "FundGuide Agent is running"}

@app.post("/check-fraud")
def check_fraud(request: FraudRequest):
    # type=0 means TRANSFER (fixed from 1)
    features = [[0, request.amount, request.oldbalanceOrg, request.newbalanceOrig, request.oldbalanceDest, request.newbalanceDest]]
    prediction = fraud_model.predict(features)[0]
    probability = fraud_model.predict_proba(features)[0][1] 
    
    if prediction == 1 or probability > 0.15:
        return {"status": "FRAUD ALERT", "is_fraud": True, "risk_score": round(probability * 100, 2), "message": "⚠️ This transaction pattern looks highly suspicious. Please verify the recipient."}
    
    return {"status": "Safe", "is_fraud": False, "risk_score": round(probability * 100, 2), "message": "✅ Transaction looks safe."}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    message_lower = request.message.lower()

    # 0. Handle general rate questions
    if "rate" in message_lower or "conversion" in message_lower:
        rate_info = get_live_usd_to_bdt()
        return ChatResponse(
            response=f"The current live exchange rate is **{rate_info['rate']:.2f} BDT** per 1 USD. (Status: {rate_info['status']} at {rate_info['timestamp']})."
        )

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

    # 3. Handle Fraud / Safety Checks
    if "safe" in message_lower or "fraud" in message_lower or "scam" in message_lower:
        numbers = re.findall(r"\d+", message_lower)
        if len(numbers) >= 2:
            amount = float(numbers[0])
            balance = float(numbers[1])
            
            # type=0 means TRANSFER (fixed from 1)
            features = [[0, amount, balance, balance - amount, 0, amount]]
            prediction = fraud_model.predict(features)[0]
            prob = fraud_model.predict_proba(features)[0][1]
            
            if prediction == 1 or prob > 0.15:
                return ChatResponse(
                    response=f"🚨 **WARNING: High Fraud Risk ({round(prob*100, 1)}%)**\n\nOur PaySim Machine Learning model has flagged this pattern. Sending **${amount:,.0f}** from a balance of **${balance:,.0f}** drains the account completely, which matches known fraud signatures. Please double-check the recipient before sending.",
                    data={"is_fraud": True, "risk_score": round(prob*100, 2)}
                )
            else:
                return ChatResponse(
                    response=f"✅ **Transaction looks safe.** Our ML model checked a transfer of ${amount:,.0f} from a balance of ${balance:,.0f} and found no suspicious patterns.",
                    data={"is_fraud": False, "risk_score": round(prob*100, 2)}
                )
        else:
            return ChatResponse(response="I can check that for you. Please tell me the amount you want to send and your current balance. For example: 'Is it safe to send 5000 from a balance of 5000?'")

    # 4. Handle Transfer Calculations & General Chat
    match_amount = re.search(r"\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)", message_lower)
    calc_data = None

    if match_amount:
        amount = int(float(match_amount.group(1).replace(",", "")))
        calc_data = compare_transfers(amount)
        
        prompt = f"""
You are a helpful remittance assistant for Bangladesh.
The user sent this exact message: "{request.message}"

Here is the exact math and data calculated by our Python engine:
{json.dumps(calc_data, indent=2)}

CRITICAL LANGUAGE INSTRUCTION:
- If the user's message is in English, reply in English.
- If the user's message uses Bengali script (বাংলা), reply in Bengali script.
- If the user's message uses Banglish (e.g., "Ami 500 dollar pathabo"), you MUST reply in Banglish.

CRITICAL CONTENT INSTRUCTIONS:
1. Explain WHY the recommended provider is best using the data.
2. You MUST mention the "family_impact" (how much more money the family gets).
3. Mention the delivery speed of the recommended provider (e.g., "Wise takes 1-2 days").
4. Do not invent any numbers. Only use the numbers provided above.
"""
    else:
        prompt = f"""
You are FundGuide AI, a helpful and polite financial assistant for Bangladesh.
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
        if calc_data:
            ai_text = f"Math Results: {calc_data['recommended']} is best. {calc_data['family_impact']}"
        else:
            ai_text = "The AI is currently busy, but our math engine is still running."

    return ChatResponse(response=ai_text, data=calc_data, action=None)