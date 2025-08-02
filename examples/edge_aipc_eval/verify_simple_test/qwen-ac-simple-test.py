import requests
import json
import datetime
import time
import os
import tqdm
import argparse

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
    print('Prompt: ', prompt)
    headers = {
        'Accept': 'text/event-stream',
        'Content-Type': 'application/json',
    }
    start_time = time.time()
    response = requests.post(url, headers=headers,
                             data=json.dumps({"prompt": prompt,"seed":10000, "stream":True}), stream=True)

    result = None
    for chunk in response.iter_content(chunk_size=1):  # chunk_size=1

        if result is None:
            result = chunk
        else:
            result += chunk
    duration = round(time.time() - start_time,2)
    result = result.decode('utf-8')

    print(' ------- ======= ------ result: ', result)

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
            print('非正常停止')
    else:
        print('请求异常')

    print('Response: ', response_content.strip())
    print('Duration: ', duration)
    return response_content.strip(), duration
    # return 'a', 1

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
                print('================================================')
                output_text, duration = generate(item['origin_prompt'], port, model)
                item['prediction'] = output_text
                item['response_time'] = duration
            save_result(data, output_dir + '/' + filename)

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
            "prompt": "Chan was born on 7 April 1954 in British Hong Kong as Chan Kong-sang to Charles and Lee-Lee Chan, political refugees from the Chinese Civil War. In circa 1937, Chan's father, originally named Fang Daolong, briefly worked as a secret agent for Lieutenant General Dai Li, the chief spy in Kuomintang-ruled China. For fear of being arrested by the communist government, Chan's father fled to British Hong Kong in the 1940s and changed his surname from Fang to Chan. Chan was his wife Chan Lee-lee’s surname. Chan discovered his father's identity and changed his Chinese name to Fang Shilong in the late 1990s, the name he would have been named according to his kin's genealogy book. Chan spent his formative years within the grounds of the French consul's residence in the Victoria Peak, British Hong Kong, as his father worked as a cook there. Chan attended the Nah-Hwa Primary School on Hong Kong Island, where he failed his first year, after which his parents withdrew him from the school. In 1960, his father emigrated to Canberra, Australia to work as the head cook for the American embassy, and Chan was sent to the China Drama Academy, a Peking Opera School run by Master Yu Jim-yuen. Chan trained rigorously for the next decade, excelling in martial arts and acrobatics. He eventually became part of the Seven Little Fortunes, a performance group made up of the school's best students, gaining the stage name Yuen Lo in homage to his master. Chan became close friends with fellow group members Sammo Hung and Yuen Biao, and the three of them later became known as the Three Brothers or Three Dragons. After entering the film industry, Chan along with Sammo Hung got the opportunity to train in hapkido under the grand master Jin Pal Kim, and Chan eventually attained a black belt. As a martial artist, Chan is also skilled in multiple forms of Kung-fu. He is also known to have trained in other martial art forms such as Karate, Judo, Taekwondo, and Jeet Kun Do.\\nQuestion: when did Chan born?Chan was born on 7 April 1954 in British Hong Kong as Chan Kong-sang to Charles and Lee-Lee Chan, political refugees from the Chinese Civil War. In circa 1937, Chan's father, originally named Fang Daolong, briefly worked as a secret agent for Lieutenant General Dai Li, the chief spy in Kuomintang-ruled China. For fear of being arrested by the communist government, Chan's father fled to British Hong Kong in the 1940s and changed his surname from Fang to Chan. Chan was his wife Chan Lee-lee’s surname. Chan discovered his father's identity and changed his Chinese name to Fang Shilong in the late 1990s, the name he would have been named according to his kin's genealogy book. Chan spent his formative years within the grounds of the French consul's residence in the Victoria Peak, British Hong Kong, as his father worked as a cook there. Chan attended the Nah-Hwa Primary School on Hong Kong Island, where he failed his first year, after which his parents withdrew him from the school. In 1960, his father emigrated to Canberra, Australia to work as the head cook for the American embassy, and Chan was sent to the China Drama Academy, a Peking Opera School run by Master Yu Jim-yuen. Chan trained rigorously for the next decade, excelling in martial arts and acrobatics. He eventually became part of the Seven Little Fortunes, a performance group made up of the school's best students, gaining the stage name Yuen Lo in homage to his master. Chan became close friends with fellow group members Sammo Hung and Yuen Biao, and the three of them later became known as the Three Brothers or Three Dragons. After entering the film industry, Chan along with Sammo Hung got the opportunity to train in hapkido under the grand master Jin Pal Kim, and Chan eventually attained a black belt. As a martial artist, Chan is also skilled in multiple forms of Kung-fu. He is also known to have trained in other martial art forms such as Karate, Judo, Taekwondo, and Jeet Kun Do.\\nQuestion: when did Chan born?"
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


# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', default='')
# parser.add_argument('--port', default='',type=str)
# parser.add_argument('--model', default='llama',type=str)
# parser.add_argument('--name', default='',type=str)
# parser.add_argument('--delay', default=0,type=int)
# args = parser.parse_args()

# time.sleep(args.delay)

# if args.name != '':
#     if not os.path.exists(f'./outputs/{args.name}'):
#         os.makedirs(f'./outputs/{args.name}')

# for d in args.dataset.split(','):
#     print(f'===== Test set: {d} =====')
#     test(d, args.port, args.model, args.name)
intit2gpu()
test2()

