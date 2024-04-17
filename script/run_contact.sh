export CUDA_VISIBLE_DEVICES=7

nohup sh run_main.sh \
      --model pretrained \
      --output_file contact-KeAP20-dmt \
      --task_name contact \
      --do_train True \
      --epoch 5 \
      --optimizer AdamW \
      --per_device_batch_size 2 \
      --gradient_accumulation_steps 8 \
      --eval_step 100 \
      --eval_batchsize 1 \
      --warmup_ratio 0.08 \
      --learning_rate 3e-5 \
      --seed 3 \
      --frozen_bert False > output/contact/KeAP20-dmt.out 2>&1

