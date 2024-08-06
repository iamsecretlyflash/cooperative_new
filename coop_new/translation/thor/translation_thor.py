# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import math
import json
import torch
import torch.nn.functional as F

from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II

from fairseq import metrics, models
from fairseq.data import encoders
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask
from fairseq.dataclass.configs import FairseqDataclass
from dataclasses import _MISSING_TYPE, dataclass, field
from typing import Any, List, Optional

import torch
from omegaconf import II, MISSING

from fairseq.dataclass.constants import (
    DATASET_IMPL_CHOICES,
    DDP_BACKEND_CHOICES,
    DDP_COMM_HOOK_CHOICES,
    GENERATION_CONSTRAINTS_CHOICES,
    GENERATION_DECODING_FORMAT_CHOICES,
    LOG_FORMAT_CHOICES,
    PIPELINE_CHECKPOINT_CHOICES,
    PRINT_ALIGNMENT_CHOICES,
    ZERO_SHARDING_CHOICES,
)

logger = logging.getLogger(__name__)


def symmetric_KL_loss(p, q, pad_mask):
    """ symmetric KL-divergence 1/2*(KL(p||q)+KL(q||p)) """
    p, q, pad_mask = p.float(), q.float(), pad_mask.view(-1)
    dict_size = q.size(-1)
    non_pad_mask = ~pad_mask
    p = p.view(-1, dict_size)[non_pad_mask]
    q = q.view(-1, dict_size)[non_pad_mask]
    loss = (p - q) * (torch.log(p) - torch.log(q))
    return 0.5 * loss.sum()


@dataclass
class TranslationThorConfig(TranslationConfig):
    num_experts: int = field(
        default=2,
        metadata={"help": "number of experts"},
    )
    consistency_alpha: float = field(
        default=1.0,
        metadata={"help": "weight of the consistency loss"},
    )
    inference_level: int = field(
        default=0,
        metadata={"help": "0 for token level, 1 for sentence level"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")

@dataclass
class GenerationConfig(FairseqDataclass):
    beam: int = field(
        default=5,
        metadata={"help": "beam size"},
    )
    beam_mt: int = field(
        default=0,
        metadata={"help": "beam size for the first-pass decoder"},
    )
    nbest: int = field(
        default=1,
        metadata={"help": "number of hypotheses to output"},
    )
    max_len_a: float = field(
        default=0,
        metadata={
            "help": "generate sequences of maximum length ax + b, where x is the source length"
        },
    )
    max_len_b: int = field(
        default=200,
        metadata={
            "help": "generate sequences of maximum length ax + b, where x is the source length"
        },
    )
    max_len_a_mt: float = field(
        default=0,
        metadata={
            "help": "generate sequences of maximum length ax + b, where x is the source length for the first-pass decoder"
        },
    )
    max_len_b_mt: int = field(
        default=200,
        metadata={
            "help": "generate sequences of maximum length ax + b, where x is the source length for the first-pass decoder"
        },
    )
    min_len: int = field(
        default=1,
        metadata={"help": "minimum generation length"},
    )
    match_source_len: bool = field(
        default=False,
        metadata={"help": "generations should match the source length"},
    )
    unnormalized: bool = field(
        default=False,
        metadata={"help": "compare unnormalized hypothesis scores"},
    )
    no_early_stop: bool = field(
        default=False,
        metadata={"help": "deprecated"},
    )
    no_beamable_mm: bool = field(
        default=False,
        metadata={"help": "don't use BeamableMM in attention layers"},
    )
    lenpen: float = field(
        default=1,
        metadata={
            "help": "length penalty: <1.0 favors shorter, >1.0 favors longer sentences"
        },
    )
    lenpen_mt: float = field(
        default=1,
        metadata={
            "help": "length penalty for the first-pass decoder: <1.0 favors shorter, >1.0 favors longer sentences"
        },
    )
    unkpen: float = field(
        default=0,
        metadata={
            "help": "unknown word penalty: <0 produces more unks, >0 produces fewer"
        },
    )
    replace_unk: Optional[str] = field(
        default=None,
        metadata={
            "help": "perform unknown replacement (optionally with alignment dictionary)",
            "argparse_const": "@@ ",
        },
    )
    sacrebleu: bool = field(
        default=False,
        metadata={"help": "score with sacrebleu"},
    )
    score_reference: bool = field(
        default=False,
        metadata={"help": "just score the reference translation"},
    )
    prefix_size: int = field(
        default=0,
        metadata={"help": "initialize generation by target prefix of given length"},
    )
    no_repeat_ngram_size: int = field(
        default=0,
        metadata={
            "help": "ngram blocking such that this size ngram cannot be repeated in the generation"
        },
    )
    sampling: bool = field(
        default=False,
        metadata={"help": "sample hypotheses instead of using beam search"},
    )
    sampling_topk: int = field(
        default=-1,
        metadata={"help": "sample from top K likely next words instead of all words"},
    )
    sampling_topp: float = field(
        default=-1.0,
        metadata={
            "help": "sample from the smallest set whose cumulative probability mass exceeds p for next words"
        },
    )
    constraints: Optional[GENERATION_CONSTRAINTS_CHOICES] = field(
        default=None,
        metadata={
            "help": "enables lexically constrained decoding",
            "argparse_const": "ordered",
        },
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "temperature for generation"},
    )
    diverse_beam_groups: int = field(
        default=-1,
        metadata={"help": "number of groups for Diverse Beam Search"},
    )
    diverse_beam_strength: float = field(
        default=0.5,
        metadata={"help": "strength of diversity penalty for Diverse Beam Search"},
    )
    diversity_rate: float = field(
        default=-1.0,
        metadata={"help": "strength of diversity penalty for Diverse Siblings Search"},
    )
    print_alignment: Optional[PRINT_ALIGNMENT_CHOICES] = field(
        default=None,
        metadata={
            "help": "if set, uses attention feedback to compute and print alignment to source tokens "
            "(valid options are: hard, soft, otherwise treated as hard alignment)",
            "argparse_const": "hard",
        },
    )
    print_step: bool = field(
        default=False,
        metadata={"help": "print steps"},
    )
    lm_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to lm checkpoint for lm fusion"},
    )
    lm_weight: float = field(
        default=0.0,
        metadata={"help": "weight for lm probs for lm fusion"},
    )

    # arguments for iterative refinement generator
    iter_decode_eos_penalty: float = field(
        default=0.0,
        metadata={"help": "if > 0.0, it penalized early-stopping in decoding."},
    )
    iter_decode_max_iter: int = field(
        default=10,
        metadata={"help": "maximum iterations for iterative refinement."},
    )
    iter_decode_force_max_iter: bool = field(
        default=False,
        metadata={
            "help": "if set, run exact the maximum number of iterations without early stop"
        },
    )
    iter_decode_with_beam: int = field(
        default=1,
        metadata={
            "help": "if > 1, model will generate translations varying by the lengths."
        },
    )
    iter_decode_with_external_reranker: bool = field(
        default=False,
        metadata={
            "help": "if set, the last checkpoint are assumed to be a reranker to rescore the translations"
        },
    )
    retain_iter_history: bool = field(
        default=False,
        metadata={
            "help": "if set, decoding returns the whole history of iterative refinement"
        },
    )
    retain_dropout: bool = field(
        default=False,
        metadata={"help": "Use dropout at inference time"},
    )
    # temporarily set to Any until https://github.com/facebookresearch/hydra/issues/1117 is fixed
    # retain_dropout_modules: Optional[List[str]] = field(
    retain_dropout_modules: Any = field(
        default=None,
        metadata={
            "help": "if set, only retain dropout for the specified modules; "
            "if not set, then dropout will be retained for all modules"
        },
    )
    # special decoding format for advanced decoding.
    decoding_format: Optional[GENERATION_DECODING_FORMAT_CHOICES] = field(
        default=None,
        metadata={"help": "special decoding format for advanced decoding."},
    )
    no_seed_provided: bool = field(
        default=False,
        metadata={"help": "if set, dont use seed for initializing random generators"},
    )
    eos_token: Optional[str] = field(
        default=None,
        metadata={"help": "EOS token"},
    )

@register_task("translation_thor", dataclass=TranslationThorConfig)
class TranslationThorTask(TranslationTask):
    """
    Translation task for Switch Transformer models.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    cfg: TranslationThorConfig

    def __init__(self, cfg: TranslationThorConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    def build_model(self, cfg):
        model = models.build_model(cfg, self)

        if self.cfg.eval_bleu:
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )

        return model

    def _get_loss(self, sample, model, criterion, expert_num=None):
        assert hasattr(
            criterion, "compute_loss"
        ), "translation_thor task requires the criterion to implement the compute_loss() method"

        encoder_out = model.encoder(
            src_tokens=sample["net_input"]["src_tokens"],
            src_lengths=sample["net_input"]["src_lengths"],
            expert_num=expert_num,
        )
        net_output = model.decoder(
            prev_output_tokens=sample["net_input"]["prev_output_tokens"],
            encoder_out=encoder_out,
            src_lengths=sample["net_input"]["src_lengths"],
            expert_num=expert_num,
        )
        loss, nll_loss = criterion.compute_loss(model, net_output, sample, reduce=True)

        logits = net_output[0].float()
        logits = F.softmax(logits, dim=-1)

        sample_size = (
            sample["target"].size(0) if criterion.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "selection": net_output[1].get("selection", None),
        }

        return loss, logits, sample_size, logging_output

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)

        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                expert_num = None
                loss1, logits1, sample_size, logging_output1 = self._get_loss(sample, model, criterion, expert_num)

        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                expert_num = None
                loss2, logits2, sample_size, logging_output2 = self._get_loss(sample, model, criterion, expert_num)

        pad_mask = sample["target"].eq(criterion.padding_idx)
        consistency_loss = symmetric_KL_loss(logits1, logits2, pad_mask)
        loss = loss1 + loss2 + consistency_loss * self.cfg.consistency_alpha

        logging_output = {
            "loss": torch.tensor([logging_output1["loss"], logging_output2["loss"]]),
            "nll_loss": torch.tensor([logging_output1["nll_loss"], logging_output2["nll_loss"]]),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "consistency": consistency_loss.data,
        }

        if ignore_grad:
            loss *= 0

        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        # follows the reduce_metrics() function in label_smoothed_cross_entropy.py
        loss_sum = sum(log.get("consistency", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "consistency", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
