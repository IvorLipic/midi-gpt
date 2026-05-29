import torch

def apply_top_k(logits, top_k):
    if top_k is None or top_k >= logits.size(-1):
        return logits
    top_k_values, top_k_indices = torch.topk(logits, top_k)
    logits = torch.full_like(logits, float("-inf"))
    logits.scatter_(-1, top_k_indices, top_k_values)
    return logits

@torch.no_grad()
def generate(model, prompt, max_new_tokens, device, top_k=None):
    model.eval()
    
    tokens = prompt.clone().to(device)

    for _ in range(max_new_tokens):
        input_ids = tokens.unsqueeze(0)

        logits = model(input_ids)

        next_logits = apply_top_k(logits[0, -1, :], top_k)
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        tokens = torch.cat([tokens, next_token], dim=0)

    return tokens
