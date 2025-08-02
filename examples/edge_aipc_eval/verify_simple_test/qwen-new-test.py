import requests
import json
import datetime
import time
import os
import tqdm
import argparse

class MetricsTracker:
    _instance = None  # This will hold the single instance of the class

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MetricsTracker, cls).__new__(cls, *args, **kwargs)
            cls._instance.prompt_data = []
            cls._instance.predicted_data = []
            cls._instance.prompt_lengths = []  # Store prompt lengths here
        return cls._instance

    def add_metrics(self, prompt_per_second, predicted_per_second, prompt_length):
        """Add new metrics data points with prompt length."""
        self.prompt_data.append(prompt_per_second)
        self.predicted_data.append(predicted_per_second)
        self.prompt_lengths.append(prompt_length)

    def print_average(self):
        """Print the average of prompt_per_second and predicted_per_second for different prompt length ranges."""
        if not self.prompt_data or not self.predicted_data or not self.prompt_lengths:
            print("No data available to calculate averages.")
            return
        
        # 定义ANSI颜色代码
        GREEN = '\033[92m'
        BLUE = '\033[94m'
        ENDC = '\033[0m'

        print('================================================================================================')
        print(f"{BLUE}Data Points:{ENDC}")
        for i in range(len(self.prompt_data)):
            print(f"Prompt/sec: {self.prompt_data[i]:.2f}, Predicted/sec: {self.predicted_data[i]:.2f}, Prompt Length: {self.prompt_lengths[i]}")

        # Calculate averages for different prompt length ranges
        ranges = [(1, 300), (300, 700), (700, 1500), (1500, 3072), (3072, 4096)]
        range_averages = {r: ([], []) for r in ranges}

        for i, length in enumerate(self.prompt_lengths):
            for r in ranges:
                if r[0] <= length < r[1]:
                    range_averages[r][0].append(self.prompt_data[i])
                    range_averages[r][1].append(self.predicted_data[i])
                    break

        print('================================================================================================')
        print(f"{GREEN}Average Metrics by Prompt Length Range:{ENDC}")
        for r in ranges:
            avg_prompt = sum(range_averages[r][0]) / len(range_averages[r][0]) if range_averages[r][0] else 0
            avg_predicted = sum(range_averages[r][1]) / len(range_averages[r][1]) if range_averages[r][1] else 0
            print(f"Prompt Length {r[0]}-{r[1]}: Avg Prompt/sec: {avg_prompt:.2f}, Avg Predicted/sec: {avg_predicted:.2f}")

        # Print overall averages
        avg_prompt = sum(self.prompt_data) / len(self.prompt_data)
        avg_predicted = sum(self.predicted_data) / len(self.predicted_data)
        print('================================================================================================')
        print(f"{GREEN}Overall Averages:{ENDC}")
        print(f"Average prompt_per_second: {avg_prompt:.2f}")
        print(f"Average predicted_per_second: {avg_predicted:.2f}")
        print('================================================================================================')

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_result(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

def generate_prompt(origin_prompt, model):
    if model == 'llama':
        if type(origin_prompt) == list:
            prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"
            for item in origin_prompt:
                if item['role'] == 'HUMAN':
                    role='<|start_header_id|>user<|end_header_id|>\n\n'
                    prompt = prompt + role + item['prompt'] + '<|eot_id|>'
                elif item['role'] == 'BOT':
                    role='<|start_header_id|>assistant<|end_header_id|>\n\n'
                    prompt = prompt + role + item['prompt'] + '<|eot_id|>'
            prompt += '<|start_header_id|>assistant<|end_header_id|>\n\n'
        else:
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{origin_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    elif model == 'mistral':
        if type(origin_prompt) == list:
            prompt = "<s>You are a helpful assistant."
            for item in origin_prompt:
                if item['role'] == 'HUMAN':
                    role='[INST] '
                    prompt = prompt + role + item['prompt'] + ''
                elif item['role'] == 'BOT':
                    role='[/INST] '
                    prompt = prompt + role + item['prompt'] + '</s>'
            prompt += '[/INST]'
        else:
            prompt = f"<s>You are a helpful assistant.[INST] {origin_prompt}[/INST]"

    elif model == 'qwen':
        if type(origin_prompt) == list:
            prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            for item in origin_prompt:
                if item['role'] == 'HUMAN':
                    role='<|im_start|>user\n'
                    prompt = prompt + role + item['prompt'] + '<|im_end|>\n'
                elif item['role'] == 'BOT':
                    role='<|im_start|>assistant\n'
                    prompt = prompt + role + item['prompt'] + '<|im_end|>\n'
            prompt += '<|im_start|>assistant\n'
        else:
            prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"+origin_prompt+"<|im_end|>\n<|im_start|>assistant\n"
    return prompt

def generate(origin_prompt, port, model):
    url=f"http://localhost:{port}/app/v1/infer/llm/chunked"
    prompt = generate_prompt(origin_prompt,model)
    #print('Prompt: ', prompt)
    headers = {
        'Accept': 'text/event-stream',
        'Content-Type': 'application/json',
    }
    start_time = time.time()
    response = requests.post(url, headers=headers,
                             data=json.dumps({
                                "prompt": prompt,
                                "n_ctx": 4096,
                                "n_predict": 1024,
                                "seed": 10000,
                                "temp": 0,
                                "top_k":40,
                                "top_p":0.9,
                                "repeat_penalty":1.1,
                                "stream":True,
                                "rope_freq_base":1000000.0
                             }),
                             stream=True)
    
    response_content = ''
    
    if response.status_code == 200:
        duration = round(time.time() - start_time,2)
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                event = decoded_line.strip().split(':', 1)[1]
                event = event.replace('false', 'False')
                event = event.replace('true', 'True')
                d = eval(event)
                if 'timings' in d and d['timings']:
                    print(d['content'])
                    timings =  d['timings']
                    for key, value in timings.items():
                        response_content += f"{key}: {value}\n"
                    predicted_ms = timings.get("predicted_ms", 0)
                    prompt_per_second = timings.get("prompt_per_second", 0)
                    predicted_per_second = timings.get("predicted_per_second", 0)
                    prompt_length = timings.get("prompt_n", 0)
                    tracker = MetricsTracker()
                    tracker.add_metrics(prompt_per_second, predicted_per_second, prompt_length)
                    #tracker.print_average()
                    break
                else:
                    print(d['content'], end='', flush=True)
    else:
        print(f"Request failed with status code {response.status_code}")
    
    print(response_content.strip())
    total_duration = duration + float(predicted_ms) / 1000 if isinstance(predicted_ms, (int, float)) else duration
    print(f'Duration: {total_duration:.2f}')
    return response_content.strip(), duration

def parse_event_stream(data):
    events = []
    event_lines = data.strip().split('\n\n')
    for event_data in event_lines:
        event = event_data.strip().split(':', 1)[1]
        if '"status":"running"' in event:
            continue
        event = event.replace('false', 'False')
        event = event.replace('true', 'True')
        d = eval(event)
        events.append(d)
    return events



def intit2gpu():
    port = 8060
    url=f"http://localhost:{port}/app/v1/model/llm/loadTogpu"
    headers = {
        'Accept': 'text/event-stream',
        'Content-Type': 'application/json',
    }
    response = requests.patch(url, headers=headers)

def test2():

    json_data = '''
    [
        {
            "prompt": "啥是联想"
        }
    ]
    '''
    data = json.loads(json_data)

    item = data[0]
    # print(item['prompt'])
    port = 8060
    model = 'qwen'


    output_text, duration = generate(item['prompt'], port, model)
    item['prediction'] = output_text
    item['response_time'] = duration

    # print(item)

def test(dataset, port, model, name):
    if name == '':
        output_dir = f'./outputs/{dataset}'
    else:
        output_dir = f'./outputs/{name}/{dataset}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for dirpath, dirnames, filenames in os.walk('./dataset/'+dataset):
        for filename in filenames:
            sub_data_path = dirpath +  '/' + filename
            data = load_data(sub_data_path)
            for _,item in data.items():
                print('================================================================================================')
                output_text, duration = generate(item['origin_prompt'], port, model)
                item['prediction'] = output_text
                item['response_time'] = duration
            save_result(data, output_dir + '/' + filename)



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='perf_test')
parser.add_argument('--port', default='8060',type=str)
parser.add_argument('--model', default='qwen',type=str)
parser.add_argument('--name', default='result',type=str)
parser.add_argument('--delay', default=0,type=int)
args = parser.parse_args()

time.sleep(args.delay)
intit2gpu()
if args.name != '':
    if not os.path.exists(f'./outputs/{args.name}'):
        os.makedirs(f'./outputs/{args.name}')

for d in args.dataset.split(','):
    print(f'===== Test set: {d} =====')
    test(d, args.port, args.model, args.name)


tracker = MetricsTracker()
tracker.print_average()

#test1()
#test2()

