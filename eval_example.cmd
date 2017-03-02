#!/bin/bash
# @ job-name=eval
# @ initialdir=.
# @ output=%j_eval2_val.out
# @ error=%j_eval2_val.err
# @ total_tasks=1
# @ gpus_per_node=1
# @ cpus-per-task=4
# @ features=k80
# @ wall_clock_limit=1:00:00
# @ partition = debug
# @ class = debug
module load merovingian
srun merovingian360+ eval.py -- --batch_size 50 --image_size 224 --run_once --eval_data val --data_dir /gpfs/projects/bscxx/bscxxxxx/mvso_en_1200/val/ \
--cnn resnet50 --eval_dir tensorflow_training_projects/resnet50_anpnet_val_eval_2/ --num_gpus 1 \
--checkpoint_dir train_2 \
--logits_output_file tensorflow_training_projects/resnet50_anpnet_val_eval_2/logits.pickle
