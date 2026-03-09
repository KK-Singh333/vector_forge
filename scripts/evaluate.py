import requests
import time
import numpy as np
import json
import os


API_URL = "http://127.0.0.1:8069/chat" 
OUTPUT_FILE = "server_responses.txt"


TEST_DATA = [
    {"q": "What is the RERA registration number for the Max Towers project?", "keyword": "UPRERAPRJ12475"},
    {"q": "Which two global architecture firms collaborated on the design of Max Towers?", "keyword": "Gensler"},
    {"q": "What green building certification has the Max Towers building achieved?", "keyword": "LEED Platinum"},
    {"q": "What is the total super built-up area of the building?", "keyword": "559,891"},
    {"q": "How many dedicated office floors does the building contain?", "keyword": "19"},
    {"q": "What is the minimum floor-to-floor ceiling height for all the office floors?", "keyword": "4.3"},
    {"q": "What is the typical size of an office floor plate?", "keyword": "25,500"},
    {"q": "How many seats are available in the building's in-house auditorium?", "keyword": "130"},
    {"q": "What is the capacity of the overhead water tank located on the roof?", "keyword": "35,000"},
    {"q": "Which metro station is located within walking distance from the property?", "keyword": "Sector 16"},
    {"q": "What is the solar heat gain coefficient of the glass used in the building's facade?", "keyword": "0.25"},
    {"q": "What is the designated parking ratio provided for the occupants?", "keyword": "1:700"},
    {"q": "How many basement levels are specifically dedicated to car parking?", "keyword": "3"},
    {"q": "What are the primary health and wellness facilities located on Level G1?", "keyword": "swimming pool"},
    {"q": "What is the total installed capacity of the chiller system in tons of refrigeration (TR)?", "keyword": "1200"},
    {"q": "How many amenity floors are included in the overall layout?", "keyword": "4"},
    {"q": "What is the floor efficiency percentage of the workspaces?", "keyword": "60%"},
    {"q": "What was the previous name of the ownership company, Max Towers Pvt. Ltd.?", "keyword": "Wise Zone"},
    {"q": "What specific type of glass configuration is utilized to reduce energy consumption?", "keyword": "Low-E"},
    {"q": "In which specific Noida sector is the Max Towers project located?", "keyword": "16B"}
]

def run_evaluation():
    latencies = []
    top_1_hits = 0
    top_3_hits = 0

    
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    print(f"Starting evaluation of {len(TEST_DATA)} queries...\n")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for idx, item in enumerate(TEST_DATA, 1):
            question = item["q"]
            target_keyword = item["keyword"].lower()
            
            
            payload = {"user_id":1,
    "query":question,
    "k":5}
            
            start_time = time.time()
            try:
                response = requests.post(API_URL, json=payload, timeout=10)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                print(f"Error on Q{idx}: {e}")
                continue
                
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
            retrieved_chunks = data.get("sources", [])
            rank_found = -1
            for rank, chunk in enumerate(retrieved_chunks):
                if target_keyword in chunk['text'].lower():
                    rank_found = rank + 1 
                    break
            if rank_found == 1:
                top_1_hits += 1
                top_3_hits += 1
            elif 1 < rank_found <= 3:
                top_3_hits += 1
            f.write(f"--- Query {idx} ---\n")
            f.write(f"Question: {question}\n")
            f.write(f"Latency: {latency:.4f} seconds\n")
            f.write(f"Target Keyword: '{target_keyword}' | Found at Rank: {rank_found if rank_found > 0 else 'Not Found in top chunks'}\n")
            f.write(f"Server Response:\n{json.dumps(data, indent=2)}\n")
            f.write("="*50 + "\n\n")

            print(f"Q{idx} processed in {latency:.3f}s | Rank: {rank_found if rank_found > 0 else 'Miss'}")
    if latencies:
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        top_1_acc = (top_1_hits / len(TEST_DATA)) * 100
        top_3_acc = (top_3_hits / len(TEST_DATA)) * 100

        print("\n" + "="*30)
        print(" EVALUATION RESULTS ")
        print("="*30)
        print(f"Total Queries:   {len(TEST_DATA)}")
        print(f"Average Latency: {avg_latency:.3f} seconds")
        print(f"P95 Latency:     {p95_latency:.3f} seconds")
        print(f"Top-1 Accuracy:  {top_1_acc:.1f}%")
        print(f"Top-3 Accuracy:  {top_3_acc:.1f}%")
        print(f"\nDetailed responses saved to: {OUTPUT_FILE}")
    else:
        print("\nNo successful requests made")

if __name__ == "__main__":
    run_evaluation()