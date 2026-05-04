# Pseudo-code for priming
def get_4_bar_prefix_remi(tokens, tokenizer, target_bars=4):
    bar_token_id = tokenizer["Bar_None"] # Check actual name in tokenizer.vocab
    bar_indices = (tokens == bar_token_id).nonzero(as_tuple=True)[0]
    
    if len(bar_indices) >= target_bars:
        cutoff = bar_indices[target_bars - 1]
        return tokens[:cutoff + 1]
    return tokens # Return all if less than 4 bars

# Field 0 is usually Bar. Bar 4 would be index 4 (or 3 depending on 0/1 indexing)
# You just find where the bar field changes to 5
def get_4_bar_prefix_oct(tokens):
    # tokens shape: (L, F)
    # Find index where field 0 (Bar) is 4
    mask = tokens[:, 0] <= 4 
    return tokens[mask]