import jax
import jax.numpy as jnp


import numpy as np

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

jax.distributed.initialize()



def square(x):
    return x ** 2

# Automatic differentiation
grad_square_scalar = jax.jit(jax.grad(square))

# Vectorization
grad_square_vectorized = jax.jit(jax.vmap(grad_square_scalar))


# Parallelization accross multiple devices

# Get the process index and the total number of processes
process_index = jax.process_index()
n_devices = jax.process_count()

# Create a vector of size 10 on each device
xs = jnp.arange(10 * n_devices, dtype=jnp.float32).reshape(n_devices, 10)[process_index:process_index+1]

# Parallelize the computation of the gradient of the square function
distributed_grad_square_vectorized = jax.pmap(grad_square_vectorized, axis_name='p')

# Compute the gradient of the square function on each device
local_output = distributed_grad_square_vectorized(xs)

# Gather all the results on the root process
if rank == 0:
    # Prepare a container to hold the received data from all processes
    # The size is the total number of processes times the size of each local_output
    gathered_outputs = np.empty([n_devices, *local_output.shape[1:]], dtype=local_output.dtype)
else:
    gathered_outputs = None
# Now the root process can print the combined array

# Use MPI's gather function to collect all arrays on the root process
comm.Gather(local_output, gathered_outputs, root=0)


if rank == 0:
    print(f"Total number of devices: {jax.device_count()}\n")
    print(f"Local/Devices per task: {jax.local_device_count()}\n")
    print("Gathered outputs on root process:")
    print(gathered_outputs)
