# Standard imports
import os
import torch
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import random
import torch
from collections import defaultdict
# from transformer_lens import HookedTransformer
from sae_lens import SAE, HookedSAETransformer
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.HookedTransformer import HookedTransformer
from __future__ import annotations
import itertools
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union, overload
import einops
import pandas as pd
import torch
from jaxtyping import Float, Int
from tqdm.auto import tqdm
from typing_extensions import Literal
import types
from transformer_lens.utils import Slice, SliceInput
import functools
from neel_plotly import line, imshow, scatter
import f_utils
import argparse  # Import argparse to handle command-line arguments

# Set up an argument parser to take the example number as input
parser = argparse.ArgumentParser(description="number of example to run")
parser.add_argument("--example_number", type=int, required=True, help="The example number to use.")
args = parser.parse_args()
example_number = args.example_number


torch.set_grad_enabled(False)
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

os.environ["HF_TOKEN"] = "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"
model = HookedSAETransformer.from_pretrained("google/gemma-2-2b", device = device)

# Attach the new method to the model instance
model.run_with_cache_with_extra_hook = types.MethodType(f_utils.run_with_cache_with_extra_hook, model)

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gemma-scope-2b-pt-res", # <- Release name 
    sae_id = "layer_8/width_16k/average_l0_71", # <- SAE id (not always a hook point!)
    device = device
)

# Example usage
N = 100  # Number of pairs to generate
dataset = f_utils.generate_dataset(N, example_number)

clean_pr = []
corr_pr = []
for i, (clean, corrupted) in enumerate(dataset):
    clean_pr.append(clean)
    corr_pr.append(corrupted)
    
clean_tokens = model.to_tokens(clean_pr)
corrupted_tokens = model.to_tokens(corr_pr)

_, clean_cache = model.run_with_cache(clean_tokens)

DO_SLOW_RUNS = True
ALL_HEAD_LABELS = [f"L{i}H{j}" for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)]
if DO_SLOW_RUNS:
    attn_head_out_act_patch_results = f_utils.get_act_patch_attn_head_out_by_pos(model, corrupted_tokens, clean_cache, equal_feature_metric)
    attn_head_out_act_patch_results = einops.rearrange(attn_head_out_act_patch_results, "layer pos head -> (layer head) pos")
    fig = imshow(attn_head_out_act_patch_results, 
        yaxis="Head Label", 
        xaxis="Pos", 
        x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))],
        y=ALL_HEAD_LABELS,
        title="attn_head_out Activation Patching By Pos", 
        return_fig=True)
    fig.write_image(f"results/example{example_number}/attn_head_out_act_patch_results.png")
    
    
# Assuming attn_head_out_act_patch_results is your tensor
sliced_results = attn_head_out_act_patch_results[:72, :]
# Adjust the y-axis labels for the first 72 elements
sliced_y_labels = ALL_HEAD_LABELS[:72]

# Adjust the x-axis labels for the last 7 positions
sliced_x_labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))] #[-7:]

fig = imshow(
    sliced_results, 
    yaxis="Head Label", 
    xaxis="Pos", 
    x=sliced_x_labels,
    y=sliced_y_labels,
    title="attn_head_out Activation Patching By Pos", 
    width=1000,  # Increase the width of the figure
    height=1200,  # Increase the height of the figure
    return_fig=True
)

# Optionally, you can adjust the tickfont size for better readability
fig.update_layout(
    yaxis=dict(tickfont=dict(size=10)),  # Adjust the size as needed
    xaxis=dict(tickfont=dict(size=10))   # Adjust the size as needed
)

# Save the figure
fig.write_image(f"results/example{example_number}/attn_head_out_act_patch_results_sliced.png")


# Assuming sliced_results is your sliced tensor
mean_value = sliced_results.mean().item()
std_dev = sliced_results.std().item()

# Calculate the threshold for one standard deviation away from the mean
lower_threshold = mean_value - std_dev
upper_threshold = mean_value + std_dev

# Identify the indices where the values are one standard deviation away from the mean
indices = (sliced_results < lower_threshold) | (sliced_results > upper_threshold)
y_indices, x_indices = torch.where(indices)

# Extract the corresponding y labels, x labels, and values
tuples_list = [
    (sliced_y_labels[y_idx], sliced_x_labels[x_idx], sliced_results[y_idx, x_idx].item())
    for y_idx, x_idx in zip(y_indices, x_indices)
]


# Convert the first element of each tuple in the list
converted_tuples = [convert_to_tuple(item[0]) for item in tuples_list] #[(convert_to_tuple(item[0]), item[1], item[2]) for item in tuples_list]

output_path = f"results/example{example_number}/converted_tuples.json"
with open(output_path, 'w') as f:
    json.dump(converted_tuples, f)

f_utils.save_relevant_attention_patterns(clean_cache, converted_tuples)
