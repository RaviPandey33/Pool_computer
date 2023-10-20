import jax.numpy as jnp
from skopt.space import Space
from skopt.sampler import Halton

def Halton_Sequence(n=100):
    ## Making the Halton code
    spacedim = [(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5) ]
    space = Space(spacedim)
    halton = Halton()
    
    halton_sequence = halton.generate(space, n)
    halton_sequence = jnp.array(halton_sequence)
    
    return halton_sequence