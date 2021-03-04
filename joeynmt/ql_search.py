import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from joeynmt.decoders import TransformerDecoder
from joeynmt.model import Model
from joeynmt.batch import Batch
from joeynmt.helpers import tile
import random

# Use this function to get the epsilon-random translated output sentence

def transformer_greedy(
        src_mask: Tensor, max_output_length: int, model: Model,
        encoder_output: Tensor, eps_threshold = 0.1) -> (np.array, None):
    """
    Special greedy function for transformer, since it works differently.
    The transformer remembers all previous states and attends to them.
    :param src_mask: mask for source inputs, 0 for positions after </s>
    :param max_output_length: maximum length for the hypotheses
    :param model: model to use for greedy decoding
    :param encoder_output: encoder hidden states for attention
    :param encoder_hidden: encoder final state (unused in Transformer)
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    bos_index = model.bos_index
    eos_index = model.eos_index
    batch_size = src_mask.size(0)

    # start with BOS-symbol for each sentence in the batch
    ys = encoder_output.new_full([batch_size, 1], bos_index, dtype=torch.long)

    # a subsequent mask is intersected with this in decoder forward pass
    trg_mask = src_mask.new_ones([1, 1, 1])
    if isinstance(model, torch.nn.DataParallel):
        trg_mask = torch.stack(
            [src_mask.new_ones([1, 1]) for _ in model.device_ids])

    finished = src_mask.new_zeros(batch_size).byte()


    for _ in range(max_output_length):
        with torch.no_grad():
            logits, _, _, _ = model(
                return_type="decode",
                trg_input=ys, # model.trg_embed(ys) # embed the previous tokens
                encoder_output=encoder_output,
                encoder_hidden=None,
                src_mask=src_mask,
                unroll_steps=None,
                decoder_hidden=None,
                trg_mask=trg_mask
            )
            # logits shape: batch_size * length of the sentence * total number of words in corpus.
            # Like torch.Size([141, 28, 31716])

            logits = logits[:, -1] # logits shape: batch_size * total number of words in corpus

            # Add noise: espilon random to get the next word
            sample = random.random()

            if sample > eps_threshold:

                _, next_word = torch.max(logits, dim=1) # greedy decoding: choose arg max over vocabulary in each step
                # next_word shape: batch_size

            else:
                random_index = np.random.randint(low=0, high=logits.shape[1], size = logits.shape[0])
                # get random index
                next_word = logits[range(logits.shape[0]), random_index]

            next_word = next_word.data.long()
            # get the word's index in corpus, return a tensor. shape: (batch_size,)
            ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
            # concatenate the next_word into a new tensor, shape: (batch_size, length of the prefix)

        # check if previous symbol was <eos>
        is_eos = torch.eq(next_word, eos_index)
        finished += is_eos

        # stop predicting if <eos> reached for all elements in batch
        if (finished >= 1).sum() == batch_size:
            break

    #ys = ys[:, 1:]  # remove BOS-symbol
    return ys.detach().cpu().numpy(), None


def push_sample_to_memory(eos_index, memory, src_batch, trans_output_batch, reward_batch, max_output_length):
    """
    Get prefix from translation output and

        :param src_mask: mask for source inputs, 0 for positions after </s>
        :param max_output_length: maximum length for the hypotheses
        :param model: model to use for greedy decoding
        :param encoder_output: encoder hidden states for attention
        :param encoder_hidden: encoder final state (unused in Transformer)
        :return:
            - stacked_output: output hypotheses (2d array of indices),
            - stacked_attention_scores: attention scores (3d array)
    """
    batch_size = src_batch.shape[0]
    for i in range(batch_size):
        source = src_batch[i]
        output_sentence = trans_output_batch[i]


        for j in range(max_output_length):

            prefix = output_sentence[:j+1]
            next_word = output_sentence[j+1]
            next_word = torch.from_numpy(np.array(next_word))
            eos_index = torch.from_numpy(np.array(eos_index))
            # check if next symbol was <eos>
            is_eos = torch.eq(next_word, eos_index)

            if not is_eos:
                memory.push(source, prefix, next_word, 0)
            else:
                memory.push(source, prefix, next_word, reward_batch[i])






