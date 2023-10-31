import numpy as np

def distance(p):
    b = p[:-1] - p[1:]
    return np.sqrt(np.sum(np.square(b)))

def angle(p):
    b = p[:-1] - p[1:]
    x = np.dot(-b[0], b[1]) / np.linalg.norm(b[0]) / np.linalg.norm(b[1])
    return np.degrees(np.arccos(x))

def dihedral(p):
    b = p[:-1] - p[1:]
    b[0] *= -1
    v = np.array( [ v - (v.dot(b[1])/b[1].dot(b[1])) * b[1] for v in [b[0], b[2]] ] )
    # Normalize vectors
    v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1,1)
    b1 = b[1] / np.linalg.norm(b[1])
    x = np.dot(v[0], v[1])
    m = np.cross(v[0], b1)
    y = np.dot(m, v[1])
    return np.degrees(np.arctan2( y, x ))