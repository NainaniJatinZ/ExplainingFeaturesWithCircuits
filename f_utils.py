import random
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
import os
import plotly.express as px

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

def generate_example_pair_0():
    # Generate two random numbers between 1 and 3 digits
    num1 = random.randint(10, 99)
    num2 = random.randint(10, 99)
    
    # Create clean and corrupted examples
    clean_example = f'What is the output of {num1} plus {num2} ? '
    corrupted_example = f'What is the output of {num1} and {num2} ? '
    
    return clean_example, corrupted_example

def generate_example_pair_1():
    # Generate two random variables with a letter followed by a digit
    num1 = random.randint(10, 99)
    num2 = random.randint(10, 99)
    var1 = chr(random.randint(97, 122)) + str(random.randint(1, 9))  # e.g., a1
    var2 = chr(random.randint(97, 122)) + str(random.randint(1, 9))  # e.g., b3
    
    # Create clean and corrupted examples
    clean_example = f'What is the output of {num1} plus {num2} ? '
    corrupted_example = f'What is the output of {var1} plus {var2} ? '
    
    return clean_example, corrupted_example


def generate_example_pair_2():
    num1 = random.randint(10, 99)
    num2 = random.randint(10, 99)
    # Generate two random variables with a letter followed by a digit
    var1 = chr(random.randint(97, 122)) + str(random.randint(1, 9))
    var2 = chr(random.randint(97, 122)) + str(random.randint(1, 9))
    
    # Create clean and corrupted examples
    clean_example = f'What is the output of {num1} plus {num2} ? '
    corrupted_example = f'What is the output of {var1} and {var2} ? '
    
    return clean_example, corrupted_example


def generate_example_pair_3():
    num1 = random.randint(10, 99)
    num2 = random.randint(10, 99)
    # Generate two random variables with a letter followed by a digit
    var1 = chr(random.randint(97, 122)) + str(random.randint(1, 9))
    var2 = chr(random.randint(97, 122)) + str(random.randint(1, 9))
    
    # Randomly choose between plus, minus, and times
    operation = random.choice(['plus', 'minus', 'times'])
    
    # Create clean and corrupted examples
    clean_example = f'What is the output of {num1} {operation} {num2} ? '
    corrupted_example = f'What is the output of {var1} and {var2} ? '
    
    return clean_example, corrupted_example

def generate_example_pair_4():
    num1 = random.randint(10, 99)
    num2 = random.randint(10, 99)
    
    # Randomly choose between plus, minus, and times
    operation = random.choice(['plus', 'minus', 'times'])
    
    # Create clean and corrupted examples
    clean_example = f'What is the output of {num1} {operation} {num2} ? '
    corrupted_example = f'What is the output of {num1} and {num2} ? '
    
    return clean_example, corrupted_example



example_dict = {0: generate_example_pair_0, 
                1: generate_example_pair_1, 
                2: generate_example_pair_2, 
                3: generate_example_pair_3, 
                4: generate_example_pair_4}
                

def generate_dataset(N, example_number):
    dataset = []
    for _ in range(N):
        if example_number in example_dict:
            clean, corrupted = example_dict[example_number]()
            dataset.append((clean, corrupted))
    return dataset


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

def equal_feature_metric(cache):
    sae_in = cache[sae.cfg.hook_name]
    feature_acts = sae.encode(sae_in)
    # print(feature_acts.shape)
    feature_acts = feature_acts.squeeze()
    return feature_acts[:, :, 15191][-2:].sum()


# Function to convert L2H1 format to (2, 1)
def convert_to_tuple(layer_head_str):
    layer = int(layer_head_str[1])
    head = int(layer_head_str[3])
    return (layer, head)


def save_relevant_attention_patterns(clean_cache, layer_head_tuples, example_number):
    # Define the directory path
    dir_path = f"results/example{example_number}/heads"
    
    # Create the directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)
    
    for layer_ind, head_ind in layer_head_tules:
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
        
        # Save the figure to the appropriate directory
        fig.write_image(f"{dir_path}/L{layer_ind}H{head_ind}_atten_pattern.png")

# def save_relevant_attention_patterns(clean_cache, layer_head_tuples, example_number):
#     for layer_ind, head_ind in layer_head_tuples:
#         temp_att_pattern = clean_cache[f'blocks.{layer_ind}.attn.hook_pattern'][1, head_ind, :, :]
#         attention_pattern = temp_att_pattern.detach().cpu().numpy()
#         # Define the x and y labels, assuming they correspond to tokens
#         tokens = model.to_str_tokens(clean_tokens[0])
#         labels = [f"{tok} {i}" for i, tok in enumerate(tokens)]

#         # Generate the heatmap
#         fig = px.imshow(
#             attention_pattern,
#             labels=dict(x="Head Position", y="Head Position", color="Attention"),
#             x=labels,
#             y=labels,
#             title=f"Attention Pattern in Layer {layer_ind}, Head {head_ind}",
#             color_continuous_scale="Blues"
#         )
#         # Display the figure
#         fig.write_image(f"example{example_number}/heads/L{layer_ind}H{head_ind}_atten_pattern.png")
        
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