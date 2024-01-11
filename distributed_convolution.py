import jax
import numpy as np
import jax.numpy as jnp

# We use mpi to gather the results in the root process
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Initialize JAX distributed required for multi-node setup
jax.distributed.initialize()

print(f"Total number of devices: {jax.device_count()}\n")
print(f"Local/Devices per task: {jax.local_device_count()}\n")

def convolve_jax(x, w):
    output = []
    for i in range(1, len(x) - 1):
        output.append(jnp.dot(x[i-1:i+2], w))
    return jnp.array(output)

# Determine the process index and the number of processes
process_index = jax.process_index()
n_devices = jax.process_count()


# We create the input arrays based on the process index
# for 8 processses the computation should yield the same result
# as in the single node case

xs = jnp.arange(5 * n_devices).reshape(n_devices, 5)[process_index:process_index+1]
ws = jnp.array([2., 3., 4.])  # Shared weights array
ws = ws.reshape(1, -1) # Ensure ws has an extra dimension to match the shape of xs

# Now we can apply pmap with the corrected data shape
distributed_convolve = jax.pmap(convolve_jax, axis_name='p')

# Each process executes its portion of the distributed convolve operation
local_output = distributed_convolve(xs, ws)

# Gather all the results on the root process
if rank == 0:
    # Prepare a container to hold the received data from all processes
    # The size is the total number of processes times the size of each local_output
    gathered_outputs = np.empty([n_devices, *local_output.shape[1:]], dtype=local_output.dtype)
else:
    gathered_outputs = None

# Use MPI's gather function to collect all arrays on the root process
comm.Gather(local_output, gathered_outputs, root=0)

# Now the root process can print the combined array
if rank == 0:
    print("Gathered outputs on root process:")
    print(gathered_outputs)
