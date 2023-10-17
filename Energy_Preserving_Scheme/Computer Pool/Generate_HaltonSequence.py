import jax.numpy as jnp
from skopt.space import Space
from skopt.sampler import Halton

def Halton_Sequence():
    ## Making the Halton code
    spacedim = [(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5) ]
    space = Space(spacedim)
    halton = Halton()
    n = 100
    
    halton_sequence = halton.generate(space, n)
    halton_sequence = jnp.array(halton_sequence)
    
    return halton_sequence