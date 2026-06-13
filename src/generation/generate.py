import torch

def apply_top_k_top_p(logits, top_k=None, top_p=None):
    """
    Filters a distribution of logits using Top-K and/or Top-P (Nucleus) filtering.
    """
    # Top-K filtering
    if top_k is not None and top_k < logits.size(-1):
        top_k_values, top_k_indices = torch.topk(logits, top_k)
        indices_to_remove = logits < top_k_values[..., -1, None]
        logits[indices_to_remove] = float("-inf")

    # Top-P (Nucleus) filtering
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)

        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float("-inf")

    return logits

@torch.no_grad()
def generate(model, prompt, tokenizer, device, temperature=1.0, top_k=None, top_p=None):
    model.eval()
    bar_token_id = tokenizer.vocab["Bar_None"]
    
    tokens = prompt.clone().to(device)
    bar_count = int((prompt == bar_token_id).sum().item())
    max_new = 1536 - len(prompt)

    for _ in range(max_new):
        input_ids = tokens.unsqueeze(0)
        logits = model(input_ids)

        next_logits = logits[0, -1, :]

        if temperature != 1.0:
            next_logits = next_logits / temperature

        next_logits = apply_top_k_top_p(next_logits, top_k, top_p)

        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == bar_token_id:
            bar_count += 1
            if bar_count >= 9:
                break

        tokens = torch.cat([tokens, next_token], dim=0)

    return tokens
