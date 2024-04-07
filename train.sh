
#!/bin/bash

python domainbed/scripts/sweep.py launch\
       --data_dir="/data/DataSets/" \
       --output_dir="/output/"\
       --command_launcher "local" \
       --algorithms GFNet_SETAug \
       --algorithms_name "GFNet_SETAug" \
       --datasets PACS \
       --datasets_name "PACS" \
       --n_hparams 1 \
       --n_trials 3 \
       --device 0 \
       --skip_confirmation \
       --single_test_envs
