MASTER_PORT=${7:-10086}


deepspeed   --hostfile  hostfile_4.txt  --master_port=${MASTER_PORT} \
    main_ds_lib_asymcheck.py  --arch  resnet152 --epochs 80  -p 100 \
       --deepspeed --deepspeed_config ds_fp16_z3_config.json \
                --multiprocessing_distributed   imagenet \

    

