import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel, PeftConfig

from utils.prompter import Prompter
import time

import json


def gen(instruction="", input_text=""):
       
    prompt = prompter.generate_prompt(instruction, input_text)
    output = pipe(prompt, max_length=512, temperature=0.2,
                  num_beams=5, eos_token_id=2)
    s = output[0]["generated_text"]
    result = prompter.get_response(s)
    return result

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    MODEL = "nlpai-lab/kullm-polyglot-12.8b-v2"
    LORA_WEIGHTS = "/workspace/my_alpaca/output_2"

    model = AutoModelForCausalLM.from_pretrained(MODEL, load_in_8bit=True, device_map={"":0})
    model = PeftModel.from_pretrained(model, LORA_WEIGHTS)
    model.eval()

    pipe = pipeline("text-generation", model=model, tokenizer=MODEL)
    prompter = Prompter("cbnu")

    filename = ""
    with open(filename, 'r') as file:
        data_test = json.load(file)
    data_test_output = list()   
    for i, d in enumerate(data_test):
        print("now " + str(i))
        try: 
            pred = gen(d['instruction'],'')
            data_test_output.append({'instruction' : d['instruction'], 'input': d['input'], 'output' : d['output'], 'pred' : pred})
        except:
            print("not output value")
            pass

    with open("", 'w') as output_file:
        json.dump(data_test_output, output_file, ensure_ascii=False, indent=4)
        print(f"The result save to {output_file}")        
