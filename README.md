# Jax on Lumi-g for the NLDL winter School


### Running Jobs Interactively

To run a job interactively on the cluster, follow these steps:

Load the required modules and environment variables:

bash```
source modules.env
````
Obtain an allocation for resources using salloc (settings are defined in salloc.sh):

bash```
sh salloc.sh
```
Run the job with srun via the appropriate shell script (e.g., `single_node_auto_diff.sh`):

bash```
sh single_node_auto_diff.sh
```
### Submitting Batch Jobs with sbatch

To submit a job as a batch process on the cluster, use the sbatch command followed by the .sbatch file associated with your program:

Submit the job to the scheduler:
bash```

sbatch single_node_auto_diff.sbatch
```
Check the output file `single_node_auto_diff.out`` for results and messages once the job completes.
