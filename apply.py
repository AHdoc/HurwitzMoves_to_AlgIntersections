import numpy, functools

def get_M(g):
    g = numpy.array(g)
    diff = g[:, None] - g[None, :]
    return numpy.where(diff == 1, 1, numpy.where(diff == -1, -1, 0))

def apply_s_to_M(s, M):
    n = M.shape[0]
    c, i = s[0], int(s[1:]) - 1
    A = [x for x in range(n) if x not in [i, i+1]]
    if c == 'R':
        M[i, A], M[i+1, A] = M[i+1, A], M[i, A] - M[i, i+1]*M[i+1, A]
        M[A, i], M[A, i+1] = M[A, i+1], M[A, i] + M[i+1, i]*M[A, i+1]
    elif c == 'L':
        M[i, A], M[i+1, A] = M[i+1, A] + M[i+1, i]*M[i, A], M[i, A]
        M[A, i], M[A, i+1] = M[A, i+1] - M[i, i+1]*M[A, i], M[A, i]
    M[i, i], M[i+1, i+1] = M[i+1, i+1], M[i, i]
    M[i, i+1], M[i+1, i] = M[i+1, i], M[i, i+1]
    return M

def sgn(x):
    return '0' if x == 0 else '+' if x > 0 else '-'

def print_M(M):
    print('M =\n' + '\n'.join([' '.join([sgn(item) for item in row]) for row in M]))

g_1=[1, 2, 3, 4, 5, 5, 4, 3, 2, 1]*2 + [1]
g_2=[1, 2, 3, 4]*5 + [1]
g_3=[1, 2, 3, 4, 5]*6 + [1]

q_1='L9  R17 L4  R3  R16 L9  L12 R10 L6  L15 R11 R12 R4  R9  R13 R13 L9  R8  R18 L19 L18 L5  R9  R12 L9  R11 R13 L14 L13 R1  L7  L17 R6  L1  L10 L5  L7  L1  R12 R4  L6  R11 L4  R8  L14 L9  R8  L18 L5  L1  L7  R4  R5  L4  L15 R3  L15 R16 L18 R2  L4  L10 L13 R6  R5  R6  R4  L9  L15 L12 R3  L6  L9  R14 R6  R16 R3  L1  R11 L12 R8  L7  L3  R13 R8  R6  L15 L2  R7  L9  R11 R17 R7  L5  L16 R11 R8  L13 R10 L2  L5'
q_2='R14 R3  L5  R4  R3  L15 L5  L17 L19 L10 R16 L6  R11 R9  L15 L12 L10 L1  L6  L13 R14 R13 L9  L15 L17 L10 L8  L14 L6  R18 L19 L18 L5  L13 R6  L12 L15 R10 R4  R13 L8  L18 R9  R4  L7  L16 R3  R9  R11 L16 R5  R3  R6  R2'
q_3='L9  L17 L11 L3  L15 L4  R25 R26 R25 L16 R4  L20 L19 L15 L16 L16 L8  R24 L1  L27 L7  R9  L10 L9  L12 L18 R17 L4  R5  R16 R23 L26 L25 R8  R1  R7  R7  R3  L14 R2  R9  R18 L20 R23 R5  R4  R22 R13 L15 L4  L20 L2  L24 R6  R6  L1  L21 L5  R23 L24 L22 L23 L19 L17 L6  L9  L15 L23 L29 R21 L29 L7  R24 R15 R22 L29 R7  L21 L4  L22 R6  R25 R20 L2  L24 L22 R6  L19 R9  R1  L26 R5  L4  R12 L10 R7  R18 L10 R27 L5  R11 L3  L6  R23 R8  L9  R4  R2  R1  R27 R28 L29 L9  L19 L12 L11 R1  R17 R28 L29 R27 L28 L27 L16 R3  L7  L29 L28 R15 R17 L13 R18 L26 R7  L8  R1  L21 L20 R16 R25 L8  L2  R21'

print_M(functools.reduce(lambda M, s: apply_s_to_M(s, M), q_1.split(), get_M(g_1)))
print_M(functools.reduce(lambda M, s: apply_s_to_M(s, M), q_2.split(), get_M(g_2)))
print_M(functools.reduce(lambda M, s: apply_s_to_M(s, M), q_3.split(), get_M(g_3)))

