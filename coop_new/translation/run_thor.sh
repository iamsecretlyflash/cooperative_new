TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.en-de \
    --workers 20

CUDA_VISIBLE_DEVICES=2 fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --max-epoch 50

fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/iwslt14.tokenized.en-de \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --max-epoch 50 \
    --save-dir 'checkpoint_cooperative2' \
    --cooperative-modules 'query,key,value,attention_out' \
    --num-experts 4

CUDA_VISIBLE_DEVICES=2 fairseq-train \
    data-bin/iwslt14.tokenized.en-de \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --max-epoch 50 \
    --save-dir 'checkpoint2'

fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe

fairseq-generate data-bin/iwslt14.tokenized.en-de \
    --path checkpoint2/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe

CUDA_VISIBLE_DEVICES=0 python train.py \
    data-bin/iwslt14.tokenized.de-en \
    --task translation_thor --user-dir thor \
    --num-experts 2 --consistency-alpha 5.0 --inference-level 1 \
    --arch thor_transformer_iwslt_de_en \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --log-format json --log-interval 10 \
    --num-workers 10 \
    --update-freq 8 \
    --ddp-backend no_c10d \
    --fp16 \
    --seed 1 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir 'thor_checkpoints' \
    --max-epoch 50

CUDA_VISIBLE_DEVICES=0 python train.py \
    data-bin/iwslt14.tokenized.en-de \
    --task translation_thor --user-dir thor \
    --num-experts 2 --consistency-alpha 5.0 --inference-level 1 \
    --arch thor_transformer_iwslt_de_en \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --log-format json --log-interval 10 \
    --num-workers 10 \
    --update-freq 8 \
    --ddp-backend no_c10d \
    --fp16 \
    --seed 1 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir 'thor_checkpoints2' \
    --max-epoch 50

fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path thor_checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    --task translation_thor --user-dir thor \
    --arch thor_transformer_iwslt_de_en

fairseq-generate data-bin/iwslt14.tokenized.en-de \
    --path thor_checkpoints2/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    --task translation_thor --user-dir thor \
    --arch thor_transformer_iwslt_de_en