from typing import Optional, Union
import logging
import os
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# Conditional AutoTokenizer import
try:
    if os.environ.get("DATASET_SOURCE", "") == "ModelScope":
        from modelscope import AutoTokenizer
        logging.info("Using ModelScope AutoTokenizer")
    else:
        from transformers import AutoTokenizer
        logging.info("Using HuggingFace transformers AutoTokenizer")
except ImportError:
    from transformers import AutoTokenizer
    logging.warning("Failed to import ModelScope AutoTokenizer, falling back to HuggingFace transformers")

"""
Note:
    Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/tokenizer.py
"""

def get_cached_tokenizer(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Get tokenizer with cached properties.

    This will patch the tokenizer object in place.

    By default, transformers will recompute multiple tokenizer properties
    each time they are called, leading to a significant slowdown. This
    function caches these properties for faster access."""

    tokenizer_all_special_ids = set(tokenizer.all_special_ids)
    tokenizer_all_special_tokens_extended = (
        tokenizer.all_special_tokens_extended)
    tokenizer_all_special_tokens = set(tokenizer.all_special_tokens)
    tokenizer_len = len(tokenizer)

    class CachedTokenizer(tokenizer.__class__):  # type: ignore

        @property
        def all_special_ids(self):
            return tokenizer_all_special_ids

        @property
        def all_special_tokens(self):
            return tokenizer_all_special_tokens

        @property
        def all_special_tokens_extended(self):
            return tokenizer_all_special_tokens_extended

        def __len__(self):
            return tokenizer_len

    CachedTokenizer.__name__ = f"Cached{tokenizer.__class__.__name__}"

    tokenizer.__class__ = CachedTokenizer
    return tokenizer


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = True,
    tokenizer_revision: Optional[str] = None,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:

    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError(
                "Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
            **kwargs)
    except ValueError as e:
        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        if (not trust_remote_code and
            ("does not exist or is not currently imported." in str(e)
             or "requires you to execute the tokenizer file" in str(e))):
            err_msg = (
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e
    except AttributeError as e:
            raise e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        logging.warning(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead.")
    return get_cached_tokenizer(tokenizer)
