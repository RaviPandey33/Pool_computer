import Transformation_Functions as TFunctions
import jax.numpy as jnp

def Convert(B1, B2, A1, A2):
    A1D = TFunctions.One_Dim_Matrix(A1)
    A1D = TFunctions.Add_B_tomatrix_A(A1D, B1)
    A2D = TFunctions.One_Dim_Matrix(A2)
    A2D = TFunctions.Add_B_tomatrix_A(A2D, B2)
    A1D = TFunctions.Add_B_tomatrix_A(A1D, A2D)
    
    return A1D