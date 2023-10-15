import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig

from utils.prompter import Prompter

MODEL = "EleutherAI/polyglot-ko-12.8b"
LORA_WEIGHTS: str = ""

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device=f"cuda", non_blocking=True)
model = PeftModel.from_pretrained(model, lora_weights)
model.eval()

pipe = pipeline("text-generation", model=model, tokenizer=MODEL, device=0)

prompter = Prompter("cbnu2")

def infer(instruction="", input_text=""):
    prompt = prompter.generate_prompt_tag(1, instruction, input_text)
    output = pipe(prompt, max_length=512, temperature=0.2, num_beams=5, eos_token_id=2)
    s = output[0]["generated_text"]
    result = prompter.get_response(s)

    return result
