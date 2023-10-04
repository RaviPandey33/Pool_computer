import jax.numpy as jnp

 # S.No. 1
def F(Bi, hBi, Aij, hAij):
    return jnp.sum(Bi)

# S.No. 2
def FyF(Bi, hBi, Aij, hAij):
    return jnp.sum(jnp.dot(Bi, Aij))


# S.No. 3
def FzG(Bi, hBi, Aij, hAij):
    return jnp.sum(jnp.dot(Bi, hAij))

# S.No. 4
def FyyFF(Bi, hBi, Aij, hAij):
    ## find summation Bi Aij Aik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += Bi[i] * Aij[i, j] * Aij[i, k]
    return result

# S.No. 5
def FyzFG(Bi, hBi, Aij, hAij):
    ## find summation Bi Aij hAik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += Bi[i] * Aij[i, j] * hAij[i, k]
    return result

# S.No. 6
def FzzGG(Bi, hBi, Aij, hAij):
    ## find summation Bi hAij hAik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += Bi[i] * hAij[i, j] * hAij[i, k]
    return result

# S.No. 7

def FyFyF(Bi, hBi, Aij, hAij):
    ## find summation Bi hAij hAik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += Bi[i] * Aij[i, j] * Aij[j, k]
    return result

# S.No. 8
def FyFzG(Bi, hBi, Aij, hAij):
    ## find summation Bi hAij hAik
    
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += Bi[i] * Aij[i, j] * hAij[j, k]
    return result

# S.No. 9
def FzGyF(Bi, hBi, Aij, hAij):
    ## find summation Bi hAij hAik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += Bi[i] * hAij[i, j] * Aij[j, k]
    return result

# S.No. 10
def FzGzG(Bi, hBi, Aij, hAij):
    ## find summation Bi hAij hAik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += Bi[i] * hAij[i, j] * hAij[j, k]
    return result


# S.No. 11 
def G(Bi, hBi, Aij, hAij):
    return jnp.sum(hBi)


# S.No. 12
def GyF(Bi, hBi, Aij, hAij):
    return jnp.sum(jnp.dot(hBi, hAij))

# S.No. 13
def GzF(Bi, hBi, Aij, hAij):
    return jnp.sum(jnp.dot(hBi, hAij))

# S.No. 14
def GzzGG(Bi, hBi, Aij, hAij):
    ## find summation Bi Aij Aik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += hBi[i] * hAij[i, j] * hAij[i, k]
    return result

# S.No. 15
def GzyGF(Bi, hBi, Aij, hAij):
    ## find summation Bi Aij hAik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += hBi[i] * hAij[i, j] * Aij[i, k]
    return result

# S.No. 16
def GyyFF(Bi, hBi, Aij, hAij):
    ## find summation Bi hAij hAik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += hBi[i] * Aij[i, j] * Aij[i, k]
    return result

# S.No. 17

def GzGzG(Bi, hBi, Aij, hAij):
    ## find summation Bi hAij hAik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += hBi[i] * hAij[i, j] * hAij[j, k]
    return result

# S.No. 18
def GyGzF(Bi, hBi, Aij, hAij):
    ## find summation Bi hAij hAik
    
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += hBi[i] * hAij[i, j] * Aij[j, k]
    return result

# S.No. 19
def GyFzG(Bi, hBi, Aij, hAij):
    ## find summation Bi hAij hAik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += hBi[i] * Aij[i, j] * hAij[j, k]
    return result

# S.No. 20
def GzFzF(Bi, hBi, Aij, hAij):
    ## find summation Bi hAij hAik
    result = 0

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result += hBi[i] * Aij[i, j] * Aij[j, k]
    return result
