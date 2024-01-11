import jax.numpy as jnp
import jax

print(f"Total number of devices: {jax.device_count()}\n")
print(f"Local/Devices per task: {jax.local_device_count()}\n")

def convolve_jax(x, w):
    output = []
    for i in range(1, len(x) - 1):
        output.append(jnp.dot(x[i-1:i+2], w))
    return jnp.array(output)

x = jnp.arange(5)
w = jnp.array([2., 3., 4.])

result_convolve_jax = convolve_jax(x, w)
print(result_convolve_jax)

### With jax.numpy and Just in time compilation

convolve_jax_jit = jax.jit(convolve_jax)

result_convolve_jax_jit = convolve_jax_jit(x, w)
print(result_convolve_jax_jit)

### With pmap

n_devices = jax.local_device_count()
xs = jnp.arange(5 * n_devices).reshape(-1, 5)
ws = jnp.stack([w] * n_devices)

print(f"xs={xs}")
print(f"ws={ws}")

distributed_convolve = jax.pmap(convolve_jax, axis_name='p')
print(xs.shape, ws.shape)
print(distributed_convolve(xs, ws))

### With pmap and Just in time compilation

n_devices = jax.local_device_count()
xs = jnp.arange(5 * n_devices).reshape(-1, 5)
ws = jnp.stack([w] * n_devices)

print(f"xs={xs}")
print(f"ws={ws}")

distributed_convolve = jax.pmap(convolve_jax_jit, axis_name='p')
print(xs.shape, ws.shape)
print(distributed_convolve(xs, ws))
