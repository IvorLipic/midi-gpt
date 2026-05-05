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
        
        L = input_ids.size(1)
        mask = torch.triu(
            torch.full((L, L), float("-inf"), device=device), 
            diagonal=1
        )

        logits = model(input_ids, attn_mask=mask)

        if isinstance(logits, list):
            next_token_fields = []
            for field_logits in logits:
                next_logits = apply_top_k(field_logits[0, -1, :], top_k)
                probs = torch.softmax(next_logits, dim=-1)
                field_sample = torch.multinomial(probs, num_samples=1)
                next_token_fields.append(field_sample)
            
            next_token = torch.cat(next_token_fields, dim=0).unsqueeze(0)
            tokens = torch.cat([tokens, next_token], dim=0)

        else:
            next_logits = apply_top_k(logits[0, -1, :], top_k)
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            tokens = torch.cat([tokens, next_token], dim=0)

    return tokens
