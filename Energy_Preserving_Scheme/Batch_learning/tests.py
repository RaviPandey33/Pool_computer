import jax.numpy as jnp

 # S.No. 1
def F(Aij, hAij, Bi, hBi):
    return jnp.sum(Bi) - 1

# S.No. 2
def FyF(Aij, hAij, Bi, hBi):
    return jnp.sum(jnp.dot(Bi, Aij)) - 2


# S.No. 3
def FzG(Aij, hAij, Bi, hBi):
    return jnp.sum(jnp.dot(Bi, hAij)) - 2 

# S.No. 4
def FyyFF(Aij, hAij, Bi, hBi):
    ## find summation Bi Aij Aik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += Bi[i] * Aij[i, j] * Aij[i, k]
    return result - 3

# S.No. 5
def FyzFG(Aij, hAij, Bi, hBi):
    ## find summation Bi Aij hAik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += Bi[i] * Aij[i, j] * hAij[i, k]
    return result - 3

# S.No. 6
def FzzGG(Aij, hAij, Bi, hBi):
    ## find summation Bi hAij hAik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += Bi[i] * hAij[i, j] * hAij[i, k]
    return result - 3

# S.No. 7

def FyFyF(Aij, hAij, Bi, hBi):
    ## find summation Bi hAij hAik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += Bi[i] * Aij[i, j] * Aij[j, k]
    return result - 6

# S.No. 8
def FyFzG(Aij, hAij, Bi, hBi):
    ## find summation Bi hAij hAik
    
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += Bi[i] * Aij[i, j] * hAij[j, k]
    return result - 6

# S.No. 9
def FzGyF(Aij, hAij, Bi, hBi):
    ## find summation Bi hAij hAik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += Bi[i] * hAij[i, j] * Aij[j, k]
    return result - 6

# S.No. 10
def FzGzG(Aij, hAij, Bi, hBi):
    ## find summation Bi hAij hAik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += Bi[i] * hAij[i, j] * hAij[j, k]
    return result - 6


# S.No. 11 
def G(Aij, hAij, Bi, hBi):
    return jnp.sum(hBi) - 1


# S.No. 12
def GyF(Aij, hAij, Bi, hBi):
    return jnp.sum(jnp.dot(hBi, hAij)) - 2

# S.No. 13
def GzF(Aij, hAij, Bi, hBi):
    return jnp.sum(jnp.dot(hBi, hAij)) - 2

# S.No. 14
def GzzGG(Aij, hAij, Bi, hBi):
    ## find summation Bi Aij Aik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += hBi[i] * hAij[i, j] * hAij[i, k]
    return result - 3 

# S.No. 15
def GzyGF(Aij, hAij, Bi, hBi):
    ## find summation Bi Aij hAik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += hBi[i] * hAij[i, j] * Aij[i, k]
    return result - 3

# S.No. 16
def GyyFF(Aij, hAij, Bi, hBi):
    ## find summation Bi hAij hAik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += hBi[i] * Aij[i, j] * Aij[i, k]
    return result - 3

# S.No. 17

def GzGzG(Aij, hAij, Bi, hBi):
    ## find summation Bi hAij hAik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += hBi[i] * hAij[i, j] * hAij[j, k]
    return result - 6

# S.No. 18
def GyGzF(Aij, hAij, Bi, hBi):
    ## find summation Bi hAij hAik
    
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += hBi[i] * hAij[i, j] * Aij[j, k]
    return result - 6

# S.No. 19
def GyFzG(Aij, hAij, Bi, hBi):
    ## find summation Bi hAij hAik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += hBi[i] * Aij[i, j] * hAij[j, k]
    return result - 6

# S.No. 20
def GzFzF(Aij, hAij, Bi, hBi):
    ## find summation Bi hAij hAik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += hBi[i] * Aij[i, j] * Aij[j, k]
    return result - 6
