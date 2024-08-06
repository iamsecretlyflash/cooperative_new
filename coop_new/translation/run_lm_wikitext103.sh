CUDA_VISIBLE_DEVICES=2 fairseq-train --task language_modeling \
  data-bin/wikitext-103 \
  --save-dir checkpoints_transformer_wiki103 \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 2048 --update-freq 16 \
  --max-update 50000 \
  --max-epoch 30 \
  --cooperative-modules ''

CUDA_VISIBLE_DEVICES=2 fairseq-eval-lm data-bin/wikitext-103 \
    --path checkpoints_transformer_wiki103/checkpoint_best.pt \
    --batch-size 2 \
    --tokens-per-sample 512 \
    --context-window 400

CUDA_VISIBLE_DEVICES=2 fairseq-train --task language_modeling \
  data-bin/wikitext-103 \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 2048 --update-freq 16 \
  --max-update 50000 \
  --max-epoch 30 \
  --save-dir 'checkpoint_cooperative_with_loss_wikitext' \
  --cooperative-modules 'intermediate, output' \
  --num-experts 8 \
  --use-entropy

CUDA_VISIBLE_DEVICES=2 fairseq-eval-lm data-bin/wikitext-103 \
    --path checkpoint_cooperative_with_loss_wikitext/checkpoint_best.pt \
    --batch-size 2 \
    --tokens-per-sample 512 \
    --context-window 400

CUDA_VISIBLE_DEVICES=2 fairseq-train --task language_modeling \
  data-bin/wikitext-103 \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 2048 --update-freq 16 \
  --max-update 50000 \
  --max-epoch 30 \
  --save-dir 'checkpoint_cooperative_without_loss_wikitext' \
  --cooperative-modules 'intermediate, output' \
  --num-experts 8

CUDA_VISIBLE_DEVICES=2 fairseq-eval-lm data-bin/wikitext-103 \
    --path checkpoint_cooperative_without_loss_wikitext/checkpoint_best.pt \
    --batch-size 2 \
    --tokens-per-sample 512 \
    --context-window 400