qsub -I -l select=1:mem=50GB:ncpus=10:ngpus=1 -l walltime=06:00:00 -q volta_login

cd /home/svu/e1100476/Project/SSL/mae;

image="/app1/common/singularity-img/3.0.0/pytorch_2.0_cuda_12.0_cudnn8-devel_u22.04.sif"

singularity exec -e /app1/common/singularity-img/3.0.0/pytorch_2.0_cuda_12.0_cudnn8-devel_u22.04.sif bash << EOF > stdout.$PBS_JOBID 2> stderr.$PBS_JOBID

export PYTHONPATH=$PYTHONPATH:/home/svu/e1100476/Packages/lib/python3.11/site-packages

nohup python main_pretrain.py &

EOF

