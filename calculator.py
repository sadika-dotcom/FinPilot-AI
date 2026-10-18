import requests
from datetime import datetime

def get_live_usd_to_bdt():
    url = "https://open.er-api.com/v6/latest/USD"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        return {
            "rate": float(data["rates"]["BDT"]),
            "source": "Live API",
            "status": "live",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
    except Exception:
        return {"rate": 118.50, "source": "Fallback", "status": "stale", "timestamp": "N/A"}

def compare_transfers(amount_usd):
    rate_info = get_live_usd_to_bdt()
    live_rate = rate_info["rate"]
    
    # 1. Define fees and rates
    providers = {
        "Wise": {"fee": round(amount_usd * 0.01, 2), "rate": live_rate + 0.5, "speed": "1-2 days"},
        "bKash": {"fee": 5, "rate": live_rate - 0.5, "speed": "Instant"},
        "Bank": {"fee": 15, "rate": live_rate - 0.2, "speed": "2-3 days"}
    }

    results = []
    for name, details in providers.items():
        # 2. The Equation
        final_bdt = (amount_usd - details["fee"]) * details["rate"]
        
        results.append({
            "name": name,
            "fee": details["fee"],
            "rate": round(details["rate"], 2),
            "speed": details["speed"],
            "final_bdt": round(final_bdt, 2),
            "equation": f"(${amount_usd} - ${details['fee']}) × {round(details['rate'], 2)} = {round(final_bdt, 2)} BDT"
        })

    # 3. Sort for Best, Good, Worst
    results.sort(key=lambda x: x["final_bdt"], reverse=True)
    
    best = results[0]
    worst = results[-1]
    
    # 4. Family Impact
    difference = round(best["final_bdt"] - worst["final_bdt"], 2)
    family_impact = f"By choosing {best['name']} over {worst['name']}, your family receives {difference} BDT more."

    return {
        "rate_info": rate_info,
        "amount_sent": amount_usd,
        "rankings": results, # Best is index 0, Worst is index 2
        "recommended": best["name"],
        "family_impact": family_impact
    }

if __name__ == "__main__":
    print(compare_transfers(500))