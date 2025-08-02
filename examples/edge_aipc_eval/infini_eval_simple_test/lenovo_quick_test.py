import requests
import json
import time
import logging
import argparse
from typing import Union
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def generate_prompt(origin_prompt: str, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> str:
    messages = [{
        "role": "user",
        "content": origin_prompt
    }]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return prompt


def generate_test(origin_prompt, host, port, tokenizer):
    url = f"http://{host}:{port}/app/v1/infer/llm/chunked"
    prompt = generate_prompt(origin_prompt, tokenizer)
    headers = {
        'Accept': 'text/event-stream',
        'Content-Type': 'application/json',
    }
    payload = {
        "prompt": prompt,
        "stream": True,
        "n_ctx": 4096,
        "n_predict": 1024,
        "seed": 10000,
        "temp": 0,
        "top_k":40,
        "top_p":0.9,
        "repeat_penalty":1.1
    }
    
    logging.info('Sending payload to LLM server')
    print(' ------- ======= ------ payload: \n', payload)
    
    start_time = time.time()
    
    response = requests.post(url, headers=headers, data=json.dumps(payload), stream=True)

    result = None
    for chunk in response.iter_content(chunk_size=1):
        if result is None:
            result = chunk
        else:
            result += chunk
    duration = round(time.time() - start_time,2)
    result = result.decode('utf-8')

    events = parse_event_stream(result)

    response_content = ''
    if len(events) >= 2:
        if 'stop' in events[-1].keys() and events[-1]['stop'] == True:
            for i in range(len(events)-1):
                response_content = response_content + events[i]['content']
            response_content = response_content + '\n'
            timings =  events[-1]['timings']
            for key, value in timings.items():
                response_content += f"{key}: {value}\n"
        else:
            logging.warning('Abnormal termination')
    else:
        logging.error('Request exception')

    logging.info(f'Response: {response_content.strip()}')
    logging.info(f'Duration: {duration} seconds')
    print('Response: ', response_content.strip())
    print('Duration: ', duration)


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


def load_model_to_gpu(host:str, port:int) -> bool:
    url=f"http://{host}:{port}/app/v1/model/llm/loadTogpu"
    headers = {
        'Accept': 'text/event-stream',
        'Content-Type': 'application/json',
    }
    response = requests.patch(url, headers=headers)
    if response.status_code == 200:
        logging.info(f"Model loaded to GPU successfully.")
        return True
    else:
        logging.error(f"Failed to load model to GPU. Status code: {response.status_code}")
        return False


def unload_model_from_gpu(host:str, port:int) -> bool:
    url=f"http://{host}:{port}/app/v1/model/llm/unloadModel"
    headers = {
        'Accept': 'text/event-stream',
        'Content-Type': 'application/json',
    }
    response = requests.patch(url, headers=headers)
    if response.status_code == 200:
        logging.info(f"Model unloaded from GPU successfully.")
        return True
    else:
        logging.error(f"Failed to unload model from GPU. Status code: {response.status_code}")
        return False


if __name__ == "__main__":
    default_prompt = "Chan was born on 7 April 1954 in British Hong Kong as Chan Kong-sang to Charles and Lee-Lee Chan, political refugees from the Chinese Civil War. In circa 1937, Chan's father, originally named Fang Daolong, briefly worked as a secret agent for Lieutenant General Dai Li, the chief spy in Kuomintang-ruled China. For fear of being arrested by the communist government, Chan's father fled to British Hong Kong in the 1940s and changed his surname from Fang to Chan. Chan was his wife Chan Lee-lee's surname. Chan discovered his father's identity and changed his Chinese name to Fang Shilong in the late 1990s, the name he would have been named according to his kin's genealogy book. Chan spent his formative years within the grounds of the French consul's residence in the Victoria Peak, British Hong Kong, as his father worked as a cook there. Chan attended the Nah-Hwa Primary School on Hong Kong Island, where he failed his first year, after which his parents withdrew him from the school. In 1960, his father emigrated to Canberra, Australia to work as the head cook for the American embassy, and Chan was sent to the China Drama Academy, a Peking Opera School run by Master Yu Jim-yuen. Chan trained rigorously for the next decade, excelling in martial arts and acrobatics. He eventually became part of the Seven Little Fortunes, a performance group made up of the school's best students, gaining the stage name Yuen Lo in homage to his master. Chan became close friends with fellow group members Sammo Hung and Yuen Biao, and the three of them later became known as the Three Brothers or Three Dragons. After entering the film industry, Chan along with Sammo Hung got the opportunity to train in hapkido under the grand master Jin Pal Kim, and Chan eventually attained a black belt. As a martial artist, Chan is also skilled in multiple forms of Kung-fu. He is also known to have trained in other martial art forms such as Karate, Judo, Taekwondo, and Jeet Kun Do.\\nQuestion: when did Chan born?Chan was born on 7 April 1954 in British Hong Kong as Chan Kong-sang to Charles and Lee-Lee Chan, political refugees from the Chinese Civil War. In circa 1937, Chan's father, originally named Fang Daolong, briefly worked as a secret agent for Lieutenant General Dai Li, the chief spy in Kuomintang-ruled China. For fear of being arrested by the communist government, Chan's father fled to British Hong Kong in the 1940s and changed his surname from Fang to Chan. Chan was his wife Chan Lee-lee's surname. Chan discovered his father's identity and changed his Chinese name to Fang Shilong in the late 1990s, the name he would have been named according to his kin's genealogy book. Chan spent his formative years within the grounds of the French consul's residence in the Victoria Peak, British Hong Kong, as his father worked as a cook there. Chan attended the Nah-Hwa Primary School on Hong Kong Island, where he failed his first year, after which his parents withdrew him from the school. In 1960, his father emigrated to Canberra, Australia to work as the head cook for the American embassy, and Chan was sent to the China Drama Academy, a Peking Opera School run by Master Yu Jim-yuen. Chan trained rigorously for the next decade, excelling in martial arts and acrobatics. He eventually became part of the Seven Little Fortunes, a performance group made up of the school's best students, gaining the stage name Yuen Lo in homage to his master. Chan became close friends with fellow group members Sammo Hung and Yuen Biao, and the three of them later became known as the Three Brothers or Three Dragons. After entering the film industry, Chan along with Sammo Hung got the opportunity to train in hapkido under the grand master Jin Pal Kim, and Chan eventually attained a black belt. As a martial artist, Chan is also skilled in multiple forms of Kung-fu. He is also known to have trained in other martial art forms such as Karate, Judo, Taekwondo, and Jeet Kun Do.\\nQuestion: when did Chan born?"

    parser = argparse.ArgumentParser(description='Run LLM inference test')
    parser.add_argument('--prompt', type=str, default=default_prompt,
                       help='Input prompt for the LLM (optional)')
    parser.add_argument('--host', type=str, default="localhost",
                       help='Host address of the LLM server (default: localhost)')
    parser.add_argument('--port', type=int, default=8060,
                       help='Port number of the LLM server (default: 8060)')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the tokenizer model (required)')
    
    args = parser.parse_args()
    
    logging.info(f"Starting LLM test with model: {args.model_path}")
    prompt = args.prompt
    host = args.host
    port = args.port
    model_path = args.model_path
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    model_loaded = False
    try:
        if not load_model_to_gpu(host, port):
            raise RuntimeError("Failed to load model to GPU")
        model_loaded = True

        generate_test(prompt, host, port, tokenizer)
    
    except Exception as e:
        logging.error(f"Test failed with error: {str(e)}")
        raise
    finally:
        if model_loaded:
            if not unload_model_from_gpu(host, port):
                logging.error("Failed to unload model from GPU")
            logging.info("Test completed successfully")
