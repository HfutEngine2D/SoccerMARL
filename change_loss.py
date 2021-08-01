#!/usr/bin/env python
"""

"""
import glob
import os

step=5
bc_iter = 2
begin = 1
end = 10
for i in range(begin, end, step):
    os.system('./kill.sh && sleep 5') 
    checkpoint = glob.glob("/home/caprlith/log"+str(i)+"/IMPALA*/IMPALA_MultiEnv*")[0]+'/checkpoint_'+str(i)+'/checkpoint-'+str(i)
    bc_command = "python ./base_IMPALA_multi_agent.py --local-dir '~/log'"+str(i+bc_iter)+" --stop-iters "+str(i+bc_iter)+" --offense-agents 2 --defense-npcs 1 --restore "+checkpoint+" --run bc"
    print("bbbbbbbbbbbbbbbcccccccccccccc : ",bc_command)
    os.system(bc_command) # two bc
    os.system('./kill.sh && sleep 5') 
    checkpoint = glob.glob("/home/caprlith/log"+str(i+bc_iter)+"/IMPALA*/IMPALA_MultiEnv*")[0]+'/checkpoint_'+str(i+bc_iter)+'/checkpoint-'+str(i+bc_iter)
    norm_impala_command = "python ./base_IMPALA_multi_agent.py --local-dir \"~/log"+str(i+step)+"\" --stop-iters "+str(i+step)+" --offense-agents 2 --defense-npcs 1 --restore "+checkpoint
    print("impalallllllllllaaaaaaaaaaaaa : ",norm_impala_command)
    os.system(norm_impala_command)
