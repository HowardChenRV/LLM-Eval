import json
import http.client
import time

def call_api(question, api_key, max_retries=3):
    """Call API and return response with retry mechanism and response time calculation"""
    prompt = question + "\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    payload = json.dumps({
        "model": "deepseek-r1-perftest",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 32768,
        "temperature": 0.6
    })
    headers = {
        'Content-Type': "application/json",
        'Authorization': f"Bearer {api_key}"
    }
    
    for attempt in range(max_retries):
        try:
            conn = http.client.HTTPSConnection("cloud.llm-ai.com")
            start_time = time.time()
            conn.request("POST", "/maas/v1/chat/completions", payload, headers)
            res = conn.getresponse()
            print(res.status, res.reason)
            response_time = time.time() - start_time
            data = res.read().decode("utf-8")
            print(data)
            conn.close()
            
            if data.strip():
                response_json = json.loads(data)
                if "choices" in response_json and response_json["choices"]:
                    return response_json, response_time
            
            print(f"Retry {attempt + 1}: Invalid response, retrying...")
            time.sleep(1)
        except (json.JSONDecodeError, http.client.HTTPException) as e:
            print(f"Error: {e}, retrying {attempt + 1}...")
            time.sleep(1)
    
    print("Max retries reached. Skipping this question.")
    return None, None

def process_file(input_file, output_file, api_key):
    """Process input file and write formatted API response to output file, calculate token generation speed"""
    total_time = 0
    total_tokens = 0
    
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for index, line in enumerate(infile):
            entry = json.loads(line)
            question = entry.get("Problem", "")
            gold_answer = entry.get("Answer", "")
            id = entry.get("id", "")
            
            api_response, response_time = call_api(question, api_key)
            if not api_response:
                print(f"API response is None, Skipping entry {index + 1} with ID {id}")
                continue  # Skip this entry if API has no response
            
            prediction = ""
            total_token_len = 0
            input_token_len = 0
            
            if "choices" in api_response and api_response["choices"]:
                prediction = api_response["choices"][0]["message"]["content"]
            
            if "usage" in api_response:
                total_token_len = api_response["usage"].get("total_tokens", 0)
                input_token_len = api_response["usage"].get("prompt_tokens", 0)
            
            token_speed = total_token_len / response_time if response_time and total_token_len else 0
            total_time += response_time if response_time else 0
            total_tokens += total_token_len
            
            # Store only current entry
            output_entry = {
                "origin_prompt": [
                    {"role": "HUMAN", "prompt": question}
                ],
                "prediction": prediction,
                "gold": gold_answer,
                "total_token_len": total_token_len,
                "input_token_len": input_token_len,
                "response_time": response_time,
                "token_speed": token_speed
            }
            
            # Output line by line instead of accumulating entire data
            outfile.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
            outfile.flush()  # Write to file immediately to prevent data loss
            
            print(f"Processed entry {index + 1}, Token Speed: {token_speed:.2f} tokens/sec")
    
    overall_speed = total_tokens / total_time if total_time and total_tokens else 0
    print(f"Total Processing Time: {total_time:.2f} seconds")
    print(f"Overall Token Generation Speed: {overall_speed:.2f} tokens/sec")

if __name__ == "__main__":
    # API_KEY = "sk-darm6xj7xv54stfn"
    API_KEY = "sk-das5tysnwiphukwp"
    INPUT_FILE = "AIME2024.jsonl"
    OUTPUT_FILE = "prediction.jsonl"
    
    process_file(INPUT_FILE, OUTPUT_FILE, API_KEY)
