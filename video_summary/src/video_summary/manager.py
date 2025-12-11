import csv
import os
import pandas as pd
from datetime import datetime
LOG_FILE = 'system_metrics.csv'

MODEL_TIERS = [
    "gemini-2.0-flash-lite-001", 
    "gemini-2.5-flash",           
    "gemini-2.5-pro"    
]
PRICING_TABLE = {
    "gemini-2.0-flash-lite-001": {
        "input": 0.075,
        "output": 0.30   
    },
    "gemini-2.5-flash": {
        "input": 0.30,
        "output": 1.20   
    },
    "gemini-2.5-pro": {
        "input_low": 1.25,  # <= 200k context
        "input_high": 2.50, # > 200k context
        "cutoff": 200000,
        "output": 5.00      
    }
}


DAILY_TOKEN_LIMIT = 200000 # 200k tokens
COST_LIMIT = 0.1 # 0.1 USD

class AdaptiveManager:
    def __init__(self):
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'model_used', 'latency', 'user_rating', 'ai_rating', 'overall_score', 'total_tokens', 'cost_usd', 'report_path'])

    def calculate_cost(self, model_name, prompt_tokens, completion_tokens):
        clean_name = model_name.replace("gemini/", "")
        price_info = PRICING_TABLE.get(clean_name, {})
        if not price_info: return 0.0
        
        input_cost = 0.0
        output_cost = 0.0
        
        if clean_name == "gemini-2.5-pro":
            price_per_1m = price_info['input_low'] if prompt_tokens <= price_info['cutoff'] else price_info['input_high']
            input_cost = (prompt_tokens / 1_000_000) * price_per_1m
        else:
            input_cost = (prompt_tokens / 1_000_000) * price_info.get('input', 0)
            
        output_cost = (completion_tokens / 1_000_000) * price_info.get('output', 0)
        return round(input_cost + output_cost, 6)

    def log_execution(self, model_used, latency, user_rating, ai_rating, overall_score, tokens, cost, report_path):
        try:
            with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now(), model_used, latency, user_rating, 
                    ai_rating, overall_score, tokens, cost, report_path
                ])
            # Debug print
            print(f"✅ Logged to CSV. Count is now: {self.get_run_count()}")
        except Exception as e:
            print(f"Error logging: {e}")

    def get_run_count(self) -> int:
        if not os.path.exists(LOG_FILE): return 0
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                row_count = sum(1 for row in f)
                return max(0, row_count - 1)
        except:
            return 0

    def decide_model(self) -> str:
       
        print("\n========== DEBUG: Starting Decision Process ==========")
        
        if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) < 50:
            print("DEBUG: Log file missing or empty. Defaulting to Tier 0.")
            return MODEL_TIERS[0]

        try:
            df = pd.read_csv(LOG_FILE)
            if df.empty: 
                print("DEBUG: DataFrame is empty. Defaulting to Tier 0.")
                return MODEL_TIERS[0]


            run_count = len(df)
            
            raw_last_model = df['model_used'].iloc[-1]
            last_model_full = str(raw_last_model).strip() 
            
            if last_model_full not in MODEL_TIERS and "gemini" not in last_model_full:
                print(f"⚠️ WARNING: Corrupt model name '{last_model_full}' found in CSV. Resetting to default.")
                last_model_full = MODEL_TIERS[0]

            print(f"🐞 DEBUG: Current Run Count: {run_count}, Last Model: {last_model_full}")
            

            if run_count > 0 and run_count % 2 != 0:
                print(f"⚓ LOCKED: Not a decision cycle ({run_count} % 2 != 0). Maintaining {last_model_full}")
                return last_model_full


            if 'overall_score' not in df.columns:
                print("ERROR: 'overall_score' column not found in CSV! Check your CSV headers.")
                if 'user_rating' in df.columns:
                    df['overall_score'] = df['user_rating']
                else:
                    return MODEL_TIERS[0]

            df['overall_score'] = pd.to_numeric(df['overall_score'], errors='coerce')
            
            valid_ratings = df[df['overall_score'] > 0]
            if not valid_ratings.empty:
                last_valid_score = valid_ratings['overall_score'].iloc[-1]
                print(f"DEBUG: Found last valid score: {last_valid_score}")
            else:
                last_valid_score = 0
            
            satisfaction = last_valid_score / 10.0 


            if 'cost_usd' in df.columns:
                total_cost = df['cost_usd'].sum()
            else:
                total_cost = 0.0

            cost_usage_percent = min(1.0, total_cost / COST_LIMIT)
            resource_available = 1.0 - cost_usage_percent


            clean_last_model = last_model_full.replace("gemini/", "").strip()
            
            if clean_last_model in MODEL_TIERS:
                current_idx = MODEL_TIERS.index(clean_last_model)
            else:
                print(f"⚠️ WARNING: Unknown model '{clean_last_model}' in CSV. Resetting to index 0.")
                current_idx = 0

            print(f"🧠 ANALYSIS: Sat={satisfaction:.2f} | Res={resource_available:.2f} | Cost Used=${total_cost:.4f}")

            
            next_idx = current_idx

            if resource_available <= 0.02:
                print("🛑 PATH: Bankruptcy Logic Triggered")
                if current_idx > 0: next_idx -= 1 
            
            elif resource_available < 0.5:
                if satisfaction <= 0.7 and current_idx < len(MODEL_TIERS) - 1:
                    next_idx += 1
                    print("🚀 PATH: Low Resource Upgrade (Critical Failure)")
                elif satisfaction > 0.7 and current_idx > 0:
                    next_idx -= 1
                    print("📉 PATH: Low Resource Downgrade")
                else:
                    print("⚖️ PATH: Low Resource Maintain")


            else:
                if satisfaction <= 0.8 and current_idx < len(MODEL_TIERS) - 1:
                    next_idx += 1
                    print(f"✨ PATH: High Resource Upgrade Triggered! ({satisfaction} <= 0.7)")
                
                elif satisfaction > 0.8 and current_idx > 0:
                    next_idx -= 1
                    print("💰 PATH: High Resource Downgrade")
                
                else:
                    print(f"⚖️ PATH: High Resource Maintain. (Sat {satisfaction} is okay)")

            new_model = MODEL_TIERS[next_idx]
            print(f"FINAL DECISION: Switching from {clean_last_model} -> {new_model}")
            print("========================================================\n")
            return new_model

        except Exception as e:
            import traceback
            print(f"\n CRITICAL ERROR in Manager: {str(e)}")
            print(traceback.format_exc())
            print(" Defaulting to Lowest Tier due to error.\n")
            return MODEL_TIERS[0]
    
    def get_resource_status(self):
       
        if not os.path.exists(LOG_FILE):
            return 1.0, 0.0, COST_LIMIT 

        try:
            df = pd.read_csv(LOG_FILE)
            if df.empty:
                return 1.0, 0.0, COST_LIMIT
            
            total_cost = df['cost_usd'].sum()
            
            usage_ratio = total_cost / COST_LIMIT
            
            
            remaining_ratio = max(0.0, 1.0 - usage_ratio)
            
            return remaining_ratio, total_cost, COST_LIMIT
            
        except Exception as e:
            print(f"Error getting resource status: {e}")
            return 1.0, 0.0, COST_LIMIT