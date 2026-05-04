import torch

@torch.no_grad()
def generate(model, prompt, max_new_tokens, device):
    model.eval()
    
    # Ensure prompt is (L) for REMI or (L, F) for Octuple
    tokens = prompt.clone().to(device)

    for _ in range(max_new_tokens):
        # 1. Prepare input: add Batch dimension -> (1, L) or (1, L, F)
        input_ids = tokens.unsqueeze(0)
        
        # 2. Create Causal Mask (L, L)
        L = input_ids.size(1)
        mask = torch.triu(
            torch.full((L, L), float("-inf"), device=device), 
            diagonal=1
        )

        # 3. Forward pass
        logits = model(input_ids, attn_mask=mask)

        # 4. Simple Sampling 
        if isinstance(logits, list):
            # ---- OCTUPLE PATH ----
            # logits is a list of Tensors: [(1, L, V_f1), (1, L, V_f2), ...]
            next_token_fields = []
            for field_logits in logits:
                # Get last time step: (V_f)
                next_logits = field_logits[0, -1, :]
                probs = torch.softmax(next_logits, dim=-1)
                # Sample one value for this specific field
                field_sample = torch.multinomial(probs, num_samples=1)
                next_token_fields.append(field_sample)
            
            # Combine fields into a single "note" vector: (1, F)
            next_token = torch.cat(next_token_fields, dim=0).unsqueeze(0)
            # Concat to tokens along the Length dimension
            tokens = torch.cat([tokens, next_token], dim=0)

        else:
            # ---- REMI PATH ----
            # logits is (1, L, V)
            next_logits = logits[0, -1, :]
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # tokens is (L), next_token is (1)
            tokens = torch.cat([tokens, next_token], dim=0)

    return tokens