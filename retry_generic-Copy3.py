from __future__ import annotations
# Standard imports
import os
import torch
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.HookedTransformer import HookedTransformer
from neel_plotly import line, imshow, scatter
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
import re
from collections import defaultdict
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

# Imports for displaying vis in Colab / notebook

torch.set_grad_enabled(False)

# For the most part I'll try to import functions and classes near where they are used
# to make it clear where they come from.

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

import torch
from collections import defaultdict

# from transformer_lens import HookedTransformer
from sae_lens import SAE, HookedSAETransformer
os.environ["HF_TOKEN"] = "<hf token here>"
model = HookedSAETransformer.from_pretrained("google/gemma-2-2b", device = device)

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gemma-scope-2b-pt-res", # <- Release name 
    sae_id = "layer_8/width_16k/average_l0_71", # <- SAE id (not always a hook point!)
    device = device
)


import random

def generate_example_pair():
    # Generate two random numbers between 1 and 3 digits
    num1 = random.randint(10, 99)
    num2 = random.randint(10, 99)
    # Generate two random variables with a letter followed by a digit
    var1 = chr(random.randint(97, 122)).upper() + chr(random.randint(97, 122)) + str(random.randint(1, 9))
    var2 = chr(random.randint(97, 122)).upper() + chr(random.randint(97, 122)) + str(random.randint(1, 9))

    operation = random.choice(['plus', 'minus', 'times'])
    
    # Create clean and corrupted examples
    clean_example = f'What is the output of {num1} {operation} {num2} ? '
    corrupted_example = f'What is the output of {var1} and {var2} ? '
    
    return clean_example, corrupted_example

def generate_dataset(N):
    dataset = []
    for _ in range(N):
        leng = 0
        while leng!=15:
            clean, corrupted = generate_example_pair()
            leng = len(model.to_tokens(corrupted)[0])
        dataset.append((clean, corrupted))
    return dataset

# Example usage
N = 100  # Number of pairs to generate
dataset = generate_dataset(N)

# Print the dataset
for i, (clean, corrupted) in enumerate(dataset):
    print(f"Pair {i+1}:")
    print(f"  Clean:     {clean}")
    print(f"  Corrupted: {corrupted}")
    print()
    if i>10:
        break
        
clean_pr = []
corr_pr = []
for i, (clean, corrupted) in enumerate(dataset):
    clean_pr.append(clean)
    corr_pr.append(corrupted)
    
def run_model_till_feature(prompt):
    _, cache = model.run_with_cache(
            prompt, 
            stop_at_layer=sae.cfg.hook_layer + 1, 
            names_filter=[sae.cfg.hook_name])
    sae_in = cache[sae.cfg.hook_name]
    feature_acts = sae.encode(sae_in).squeeze()
    return feature_acts[:, 15191][-2:].sum()

for i, (clean, corrupted) in enumerate(dataset):
    print("Clean: ", run_model_till_feature(clean))
    print("Corrupted: ", run_model_till_feature(corrupted))

    
def run_with_cache_with_extra_hook(
    self,
    *model_args: Any,
    current_activation_name: str,
    current_hook: Any,
    names_filter: NamesFilter = None,
    device: DeviceType = None,
    remove_batch_dim: bool = False,
    incl_bwd: bool = False,
    reset_hooks_end: bool = True,
    clear_contexts: bool = False,
    pos_slice: Optional[Union[Slice, SliceInput]] = None,
    **model_kwargs: Any,
):
    """
    Runs the model and returns the model output and a Cache object.
    
    Adds an extra forward hook (current_activation_name, current_hook) to the hooks.

    Args:
        *model_args: Positional arguments for the model.
        current_activation_name: The name of the activation to hook.
        current_hook: The hook function to use.
        names_filter (NamesFilter, optional): A filter for which activations to cache.
        device (str or torch.Device, optional): The device to cache activations on.
        remove_batch_dim (bool, optional): If True, removes the batch dimension when caching.
        incl_bwd (bool, optional): If True, caches gradients as well.
        reset_hooks_end (bool, optional): If True, removes all hooks added by this function.
        clear_contexts (bool, optional): If True, clears hook contexts whenever hooks are reset.
        pos_slice: The slice to apply to the cache output. Defaults to None.
        **model_kwargs: Keyword arguments for the model.

    Returns:
        tuple: A tuple containing the model output and a Cache object.
    """

    pos_slice = Slice.unwrap(pos_slice)

    # Get the caching hooks
    cache_dict, fwd, bwd = self.get_caching_hooks(
        names_filter,
        incl_bwd,
        device,
        remove_batch_dim=remove_batch_dim,
        pos_slice=pos_slice,
    )

    # Add the extra forward hook
    fwd_hooks = [(current_activation_name, current_hook)] + fwd

    # Run the model with the hooks
    with self.hooks(
        fwd_hooks=fwd_hooks,
        bwd_hooks=bwd,
        reset_hooks_end=reset_hooks_end,
        clear_contexts=clear_contexts,
    ):
        model_out = self(*model_args, **model_kwargs)
        if incl_bwd:
            model_out.backward()

    return model_out, cache_dict



# Attach the new method to the model instance
model.run_with_cache_with_extra_hook = types.MethodType(run_with_cache_with_extra_hook, model)




def generic_activation_patch(
    model: HookedTransformer,
    corrupted_tokens: Int[torch.Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: Callable[[Float[torch.Tensor, "batch pos d_vocab"]], Float[torch.Tensor, ""]],
    patch_setter: Callable[
        [CorruptedActivation, Sequence[int], ActivationCache], PatchedActivation
    ],
    activation_name: str,
    index_axis_names: Optional[Sequence[AxisNames]] = None,
    index_df: Optional[pd.DataFrame] = None,
    return_index_df: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, pd.DataFrame]]:
    """
    A generic function to do activation patching, will be specialised to specific use cases.

    Activation patching is about studying the counterfactual effect of a specific activation between a clean run and a corrupted run. The idea is have two inputs, clean and corrupted, which have two different outputs, and differ in some key detail. Eg "The Eiffel Tower is in" vs "The Colosseum is in". Then to take a cached set of activations from the "clean" run, and a set of corrupted.

    Internally, the key function comes from three things: A list of tuples of indices (eg (layer, position, head_index)), a index_to_act_name function which identifies the right activation for each index, a patch_setter function which takes the corrupted activation, the index and the clean cache, and a metric for how well the patched model has recovered.

    The indices can either be given explicitly as a pandas dataframe, or by listing the relevant axis names and having them inferred from the tokens and the model config. It is assumed that the first column is always layer.

    This function then iterates over every tuple of indices, does the relevant patch, and stores it

    Args:
        model: The relevant model
        corrupted_tokens: The input tokens for the corrupted run
        clean_cache: The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)
        patch_setter: A function which acts on (corrupted_activation, index, clean_cache) to edit the activation and patch in the relevant chunk of the clean activation
        activation_name: The name of the activation being patched
        index_axis_names: The names of the axes to (fully) iterate over, implicitly fills in index_df
        index_df: The dataframe of indices, columns are axis names and each row is a tuple of indices. Will be inferred from index_axis_names if not given. When this is input, the output will be a flattened tensor with an element per row of index_df
        return_index_df: A Boolean flag for whether to return the dataframe of indices too

    Returns:
        patched_output: The tensor of the patching metric for each patch. By default it has one dimension for each index dimension, via index_df set explicitly it is flattened with one element per row.
        index_df *optional*: The dataframe of indices
    """

    if index_df is None:
        assert index_axis_names is not None

        # Get the max range for all possible axes
        max_axis_range = {
            "layer": model.cfg.n_layers,
            "pos": corrupted_tokens.shape[-1],
            "head_index": model.cfg.n_heads,
        }
        max_axis_range["src_pos"] = max_axis_range["pos"]
        max_axis_range["dest_pos"] = max_axis_range["pos"]
        max_axis_range["head"] = max_axis_range["head_index"]

        # Get the max range for each axis we iterate over
        index_axis_max_range = [max_axis_range[axis_name] for axis_name in index_axis_names]

        # Get the dataframe where each row is a tuple of indices
        index_df = transformer_lens.patching.make_df_from_ranges(index_axis_max_range, index_axis_names)

        flattened_output = False
    else:
        # A dataframe of indices was provided. Verify that we did not *also* receive index_axis_names
        assert index_axis_names is None
        index_axis_max_range = index_df.max().to_list()

        flattened_output = True

    # Create an empty tensor to show the patched metric for each patch
    if flattened_output:
        patched_metric_output = torch.zeros(len(index_df), device=model.cfg.device)
    else:
        patched_metric_output = torch.zeros(index_axis_max_range, device=model.cfg.device)

    # A generic patching hook - for each index, it applies the patch_setter appropriately to patch the activation
    def patching_hook(corrupted_activation, hook, index, clean_activation):
        return patch_setter(corrupted_activation, index, clean_activation)

    # Iterate over every list of indices, and make the appropriate patch!
    for c, index_row in enumerate(tqdm((list(index_df.iterrows())))):
        index = index_row[1].to_list()

        # The current activation name is just the activation name plus the layer (assumed to be the first element of the input)
        current_activation_name = utils.get_act_name(activation_name, layer=index[0])

        # The hook function cannot receive additional inputs, so we use partial to include the specific index and the corresponding clean activation
        current_hook = partial(
            patching_hook,
            index=index,
            clean_activation=clean_cache[current_activation_name],
        )
        
#         incl_bwd = False
#         cache_dict, fwd, bwd = model.get_caching_hooks(
#             incl_bwd=incl_bwd,
#             device=device,
#             names_filter=None
#         )
        
#         fwd_hooks = [(current_activation_name, current_hook)] + fwd
        # Run the model with the patching hook and get the logits!
        # patched_logits, patched_cache = "", ""
        
        patched_logits, patched_cache = model.run_with_cache_with_extra_hook(
            corrupted_tokens, 
            current_activation_name=current_activation_name, 
            current_hook= current_hook
        )
        # print(patched_cache.keys())
        # print(patched_logits.shape)

        # Calculate the patching metric and store
        if flattened_output:
            patched_metric_output[c] = patching_metric(patched_cache).item()
        else:
            patched_metric_output[tuple(index)] = patching_metric(patched_cache).item()

    if return_index_df:
        return patched_metric_output, index_df
    else:
        return patched_metric_output

def layer_pos_patch_setter(corrupted_activation, index, clean_activation):
    """
    Applies the activation patch where index = [layer, pos]

    Implicitly assumes that the activation axis order is [batch, pos, ...], which is true of everything that is not an attention pattern shaped tensor.
    """
    assert len(index) == 2
    layer, pos = index
    corrupted_activation[:, pos, ...] = clean_activation[:, pos, ...]
    return corrupted_activation
    
get_act_patch_resid_pre = partial(
    generic_activation_patch,
    patch_setter=layer_pos_patch_setter,
    activation_name="resid_pre",
    index_axis_names=("layer", "pos"),
)


def equal_feature_metric(cache):
    sae_in = cache[sae.cfg.hook_name]
    feature_acts = sae.encode(sae_in)
    # print(feature_acts.shape)
    feature_acts = feature_acts.squeeze()
    return feature_acts[:, :, 15191][-2:].sum()


clean_tokens = model.to_tokens(clean_pr)
corrupted_tokens = model.to_tokens(corr_pr)

_, clean_cache = model.run_with_cache(clean_tokens)
_, corrupted_cache = model.run_with_cache(corrupted_tokens)


resid_pre_act_patch_results = get_act_patch_resid_pre(model, corrupted_tokens, clean_cache, equal_feature_metric)

fig = imshow(
    resid_pre_act_patch_results, 
    yaxis="Layer", 
    xaxis="Position", 
    x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))],
    title="resid_pre Activation Patching",
    return_fig=True  # This ensures the figure object is returned
)

fig.write_image("results/exp3/resid_pre_activation_patching1.png")


def layer_head_vector_patch_setter(
    corrupted_activation,
    index,
    clean_activation,
):
    """
    Applies the activation patch where index = [layer,  head_index]

    Implicitly assumes that the activation axis order is [batch, pos, head_index, ...], which is true of all attention head vector activations (q, k, v, z, result) but *not* of attention patterns.
    """
    assert len(index) == 2
    layer, head_index = index
    corrupted_activation[:, :, head_index] = clean_activation[:, :, head_index]

    return corrupted_activation

get_act_patch_attn_head_out_all_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_vector_patch_setter,
    activation_name="z",
    index_axis_names=("layer", "head"),
)

attn_head_out_all_pos_act_patch_results = get_act_patch_attn_head_out_all_pos(model, corrupted_tokens, clean_cache, equal_feature_metric)
fig = imshow(attn_head_out_all_pos_act_patch_results,  
       yaxis="Layer", 
       xaxis="Head", 
       title="attn_head_out Activation Patching (All Pos)", 
        return_fig=True)

fig.write_image("results/exp3/attn_head_out Activation Patching All Pos1.png")


def layer_pos_head_vector_patch_setter(
    corrupted_activation,
    index,
    clean_activation,
):
    """
    Applies the activation patch where index = [layer, pos, head_index]

    Implicitly assumes that the activation axis order is [batch, pos, head_index, ...], which is true of all attention head vector activations (q, k, v, z, result) but *not* of attention patterns.
    """
    assert len(index) == 3
    layer, pos, head_index = index
    corrupted_activation[:, pos, head_index] = clean_activation[:, pos, head_index]
    return corrupted_activation

get_act_patch_attn_head_out_by_pos = partial(
    generic_activation_patch,
    patch_setter=layer_pos_head_vector_patch_setter,
    activation_name="z",
    index_axis_names=("layer", "pos", "head"),
)

DO_SLOW_RUNS = True
ALL_HEAD_LABELS = [f"L{i}H{j}" for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)]
if DO_SLOW_RUNS:
    attn_head_out_act_patch_results = get_act_patch_attn_head_out_by_pos(model, corrupted_tokens, clean_cache, equal_feature_metric)
    attn_head_out_act_patch_results = einops.rearrange(attn_head_out_act_patch_results, "layer pos head -> (layer head) pos")
    fig = imshow(attn_head_out_act_patch_results, 
        yaxis="Head Label", 
        xaxis="Pos", 
        x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))],
        y=ALL_HEAD_LABELS,
        title="attn_head_out Activation Patching By Pos", 
        return_fig=True)
    fig.write_image("results/exp3/attn_head_out_act_patch_results1.png")


# Assuming attn_head_out_act_patch_results is your tensor
sliced_results = attn_head_out_act_patch_results[:72, -7:]
# Adjust the y-axis labels for the first 72 elements
sliced_y_labels = ALL_HEAD_LABELS[:72]

# Adjust the x-axis labels for the last 7 positions
sliced_x_labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))][-7:]

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
fig.write_image("results/exp3/attn_head_out_act_patch_results_sliced1.png")

import torch

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

# Display the tuples
print(tuples_list)


# Function to convert L2H1 format to (2, 1)
def convert_to_tuple(layer_head_str):
    layer = int(layer_head_str[1])
    head = int(layer_head_str[3])
    return (layer, head)

# Convert the first element of each tuple in the list
converted_tuples = [convert_to_tuple(item[0]) for item in tuples_list] #[(convert_to_tuple(item[0]), item[1], item[2]) for item in tuples_list]
import json
# Display the result
print("tuples", converted_tuples)
output_path = f"results/exp3/converted_tuples1.json"
with open(output_path, 'w') as f:
    json.dump(converted_tuples, f)


def save_relevant_attention_patterns(clean_cache, layer_head_tuples):
    for layer_ind, head_ind in layer_head_tuples:
        temp_att_pattern = clean_cache[f'blocks.{layer_ind}.attn.hook_pattern'][1, head_ind, :, :]
        attention_pattern = temp_att_pattern.detach().cpu().numpy()
        # Define the x and y labels, assuming they correspond to tokens
        tokens = model.to_str_tokens(clean_tokens[0])
        labels = [f"{tok} {i}" for i, tok in enumerate(tokens)]

        # Generate the heatmap
        fig = px.imshow(
            attention_pattern,
            labels=dict(x="Head Position", y="Head Position", color="Attention"),
            x=labels,
            y=labels,
            title=f"Attention Pattern in Layer {layer_ind}, Head {head_ind}",
            color_continuous_scale="Blues"
        )
        # Display the figure
        fig.write_image(f"results/exp3/heads/L{layer_ind}H{head_ind}_atten_pattern1.png")
        
save_relevant_attention_patterns(clean_cache, converted_tuples)