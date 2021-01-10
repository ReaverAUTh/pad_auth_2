#!/bin/bash
#SBATCH --job-name=PDS
#SBATCH --time=00:35:00              
#SBATCH --partition=batch             
#SBATCH --ntasks-per-node=15
#SBATCH --nodes=2
#SBATCH --output=my_job1.stdout    

module load gcc openmpi openblas

export OMPI_MCA_btl_vader_single_copy_mechanism=none

make clean
make all

echo "Testing FMA, d=518, N=106574 || k=55, p=30"
./V0
srun -n 30 ./V1
srun -n 30 ./V2
make clean
