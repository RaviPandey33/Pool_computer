import jax.numpy as jnp

def Lobatto3A3B_4thOrder():
    # Lobatto 3A and B fourth order, is implicit method
    ## Lobatto IIIB methods are A-stable, but not L-stable and B-stable.
    A1 = jnp.array([
        [0., 0., 0., 0.],
        [5/24, 1/3, -1/24, 0.],
        [1/6, 2/3, 1/6, 0.],
        [0., 0., 0., 0.]])
    B1 = jnp.array([1/6, 2/3, 1/6, 0.])

    # Lobatto IIIB fourth-order
    A2 = jnp.array([
        [1/6, -1/6, 0., 0.],
        [1/6, 1/3, 0., 0.],
        [1/6, 5/6, 0., 0.],
        [0., 0., 0., 0.]])
    B2 = jnp.array([1/6, 2/3, 1/6, 0.])
    
    return A1, A2, B1, B2
    
    
def RK_Explicit_4thOrder():
    # Lobatto 3A and B fourth order
    A1 = jnp.array([
        [0., 0., 0., 0.],
        [1/2, 0., 0., 0.],
        [0., 1/2, 0., 0.],
        [0., 0., 1, 0.]])
    B1 = jnp.array([1/6, 1/3, 1/3, 1/6])

    # Lobatto IIIB fourth-order
    A2 = jnp.array([
        [0., 0., 0., 0.],
        [1/2, 0., 0., 0.],
        [0., 1/2, 0., 0.],
        [0., 0., 1, 0.]])
    B2 = jnp.array([1/6, 1/3, 1/3, 1/6])
    return A1, A2, B1, B2