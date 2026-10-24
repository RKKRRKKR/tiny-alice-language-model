import torch
import torch.nn.functional as F
from train import TinyGPT, encode, decode, block_size

checkpoint = torch.load("tiny_model.pt")
model = TinyGPT()
model.load_state_dict(checkpoint["state"])
model.eval()

@torch.no_grad()
def generate(prompt, n=400, temperature=0.8):
    idx = torch.tensor([encode(prompt)], dtype=torch.long)
    for _ in range(n):
        logits, _ = model(idx[:, -block_size:])
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_char = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_char], dim=1)
    return decode(idx[0].tolist())

print("Tiny Alice model ready. Type a starting phrase (or 'quit').")
while True:
    prompt = input("\nPrompt: ")
    if prompt.strip().lower() == "quit":
        break
    if not prompt:
        prompt = "\n"
    try:
        print(generate(prompt))
    except KeyError as e:
        print("That character isn't in the model's vocabulary:", e)