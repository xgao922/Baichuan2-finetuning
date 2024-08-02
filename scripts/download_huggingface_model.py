from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-9b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("THUDM/glm-4-9b", trust_remote_code=True)

tokenizer.save_pretrained("/home/xgao/Baichuan2/glm-4-9b")
model.save_pretrained("/home/xgao/Baichuan2/glm-4-9b")