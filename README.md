<p align="center">
  <img width="600" height="257" src="https://i.imgur.com/oqrOAX8.png">
</p>

# Parallel-Exercise2

Various implementations of an all-knn algorithm, using MPI and openMP.

**/data/** contains the datasets used to produce the results.

**/results/** contains the results produced by using AUTh's High Performance Computing (HPC) infrastructure. 

**/hpc_auth/** contains files used to run the implementations on the HPC.

## **1. Before using**
File *supplementary.c* contains a function called *read_from_file*. There we can set the name of the file we want to use as our data and declare its N=number_of_points and d=dimensions values.

**V0:** Serial Implementation

**V1:** Asynchronous Implementation (using MPI)

**V2:** Vantage Point Tree Implementation (MPI)

## **2. Local execution**
In order to test the implementations locally on your machine, use the files located in the home directory. Follow the commands in the order given below:

```
make clean
make all
make test p=<val>
```

Make clean should be used if you have already ran the programs before. Instead of val type the number of processes you wish to use for the V1, V2 implementations. All of the above commands are declared in the Makefile.

## **3. HPC execution**
Everything can be run on AUTh's HPC (for those with an account), by using the same files described above but by changing the Makefile with the one located in the *hpc_auth* directory. Use the shell file located in that directory as well. Edit it according to the nodes/tasks you want to use and submit it to the HPC for execution. To do so, run the following command in the shell:

```
sbatch <shell_file_name>.sh
```
\
#
Repo for the second exercise of course 050 - Parallel and Distributed Systems, Aristotle University of Thessaloniki, Dpt. of Electrical & Computer Engineering.

