#!/bin/bash
# @ job_name= train
# @ initialdir= .
# @ output= %j.out
# @ error= %j.err
# @ total_tasks= 2
# @ gpus_per_node= 4
# @ cpus_per_task= 16
# @ wall_clock_limit = 00:15:00
# @ features = k80


DIR_PATH=/gpfs/projects/bscxx/bscxxxxx/DISTRIBUTED

module load K80 cuda/8.0 mkl/2017.1 CUDNN/5.1.10-cuda_8.0 intel-opencl/2016 python/3.6.0+_ML

PARAM_SERVERS=1
NUM_GPUS=4
ALL_GPUS_IN_NODE=1
PS_ALONGSIDE=1

FILE_TASK_NAME="$SLURM_JOB_ID"
FILE_TASK_NAME+="_greasy"

TRAIN_DIR="train"

#SCRIPT TO CREATE GREASY FILE -> https://github.com/xiscosc/cluster_mt_creator/

SCRIPT="main_dist.py --sync_replicas --backup_workers 0 --batch_size 32 --evaluation_job eval_example.cmd --image_size 224 --data_dir /gpfs/projects/bscxx/bscxxxxx/DISTRIBUTED/mvso_en_1200/train/ --max_steps 1000000 --num_gpus $NUM_GPUS --cnn resnet50 --optimizer sgd --initial_learning_rate 0.1 --weight_decay_rate 0.0001 --train_dir $TRAIN_DIR --num_readers 8 --checkpoint /gpfs/projects/bscxx/bscxxxxx/DISTRIBUTED/cnn_models/resnet_v1_50_anp.ckpt"

FILE_TASKS=$DIR_PATH/job_logs/$FILE_TASK_NAME.txt
export GREASY_LOGFILE=$DIR_PATH/job_logs/$FILE_TASK_NAME.log

python mt_cluster.py $FILE_TASKS "$SCRIPT" $PARAM_SERVERS $NUM_GPUS $ALL_GPUS_IN_NODE $PS_ALONGSIDE
/apps/GREASY/latest/bin/greasy $FILE_TASKS
