python -m openmatch.driver.train_dr  \
    --output_dir $CHECKPOINT_DIR/clue_cos_t5/t5  \  # for checkpoints
    --model_name_or_path $PLM_DIR/t5-base-scaled  \  # HF PLM
    --do_train  \
    --save_steps 20000  \
    --eval_steps 20000  \
    --train_path $PROCESSED_DIR/msmarco/t5/train.new.jsonl  \
    --eval_path $PROCESSED_DIR/msmarco/t5/val.jsonl  \
    --fp16  \  # recommended
    --per_device_train_batch_size 8  \
    --train_n_passages 8  \
    --learning_rate 5e-6  \
    --q_max_len 32  \
    --p_max_len 128  \
    --num_train_epochs 3  \
    --logging_dir $LOG_DIR/msmarco/t5  \  # tensorboard logging dir
    --evaluation_strategy steps  # evaluate every `eval_steps` steps


python -m openmatch.driver.train_dr  \
    --output_dir $CHECKPOINT_DIR/msmarco/t5  \  # for checkpoints
    --model_name_or_path $PLM_DIR/t5-base-scaled  \  # HF PLM
    --do_train  \
    --save_steps 20000  \
    --eval_steps 20000  \
    --train_path $PROCESSED_DIR/msmarco/t5/train.new.jsonl  \
    --eval_path $PROCESSED_DIR/msmarco/t5/val.jsonl  \
    --fp16  \  # recommended
    --per_device_train_batch_size 8  \
    --train_n_passages 8  \
    --learning_rate 5e-6  \
    --q_max_len 32  \
    --p_max_len 128  \
    --num_train_epochs 3  \
    --logging_dir $LOG_DIR/msmarco/t5  \  # tensorboard logging dir
    --evaluation_strategy steps  # evaluate every `eval_steps` steps




python /home/jingyuah/AnchorDR/lib/openmatch/driver/train_dr.py \
    --output_dir $CHECKPOINT_DIR/clue_cos_t5/t5_s2 \
    --model_name_or_path $CHECKPOINT_DIR/clue_cos_t5/t5  \  # start from stage 1 checkpoint
    --do_train  \
    --save_steps 20000  \
    --eval_steps 20000  \
    --corpus_path $path_to_corpus \
    --doc_template "Title: <title> Text: <text>" \
    --doc_column_names id,title,text \
    --q_max_len 32 \
    --p_max_len 128 \
    --fp16 \
    --dataloader_num_workers 1 \
    # --use_viz_features $viz_features \
    2>&1 &

python -m openmatch.driver.train_dr  \
    --output_dir $CHECKPOINT_DIR/msmarco/t5_s2  \
    --model_name_or_path $CHECKPOINT_DIR/msmarco/t5  \  # start from stage 1 checkpoint
    --do_train  \
    --save_steps 20000  \
    --eval_steps 20000  \
    --train_path $PROCESSED_DIR/msmarco/t5/train.new.hn.jsonl  \
    --eval_path $PROCESSED_DIR/msmarco/t5/val.hn.jsonl  \
    --fp16  \
    --per_device_train_batch_size 8  \
    --train_n_passages 8  \
    --learning_rate 5e-6  \
    --q_max_len 32  \
    --p_max_len 128  \
    --num_train_epochs 3  \
    --logging_dir $LOG_DIR/msmarco/t5_s2  \
    --evaluation_strategy steps