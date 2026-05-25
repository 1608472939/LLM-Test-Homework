from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "/mnt/data/Qwen-7B-Chat"
prompt = "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto"
).eval()

response, history = model.chat(tokenizer, prompt, history=None)
print("Qwen大模型的回答：\n", response)