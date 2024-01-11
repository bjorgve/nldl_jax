import jax
import jax.numpy as jnp

print(f"Total number of devices: {jax.device_count()}\n")
print(f"Local/Devices per task: {jax.local_device_count()}\n")

# Define a simple function that computes the square of its input.
def square(x):
    return x ** 2

# AUTODIFF:
# Compute the gradient function for the scalar `square` function.
grad_square_scalar = jax.jit(jax.grad(square))

# VECTORIZATION:
# Vectorize the gradient computation to work element-wise on arrays.
grad_square_vectorized = jax.jit(jax.vmap(grad_square_scalar))

# Evaluate the gradient for a scalar input, x = 3.0, using JIT compilation.
x = 3.0
gradient_scalar = grad_square_scalar(x)
# Print the result with a clear descriptive message.
print("\n" * 3)
print(f'Gradient of square at x = {x}: {gradient_scalar} (Should be 6.0, as the derivative of x^2 is 2x)')

# Evaluate the gradient at multiple points using vectorization and JIT.
x = jnp.arange(5, dtype=jnp.float32)
gradient = grad_square_vectorized(x)
# Print the gradient for each element within the array with a descriptive message.
print("\n" * 3)
print(f'Gradient of square at points {x.tolist()}: {gradient.tolist()} (Element-wise gradient of x^2)')


print("\n")
print(f"Total number of devices: {jax.device_count()}")
print(f"Local/Devices per task: {jax.local_device_count()}\n")

n_devices = jax.local_device_count()
xs = jnp.arange(10 * n_devices, dtype=jnp.float32).reshape(-1, 10)
print(f'Data prepared for distribution across {n_devices} devices:\n{xs}')

distributed_grad_square_vectorized = jax.pmap(grad_square_vectorized, axis_name='p')

distributed_gradients = distributed_grad_square_vectorized(xs)
print("\n")
print(f'Distributed gradients:\n{distributed_gradients}')
