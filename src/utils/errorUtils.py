"""
Provides an abstraction over the different methods of error propagation during
mathematical calculations
"""

import math

try:
    import soerp as so
    import soerp.umath as so_math
    _HAS_SOERP = True
except Exception:
    so = None
    so_math = None
    _HAS_SOERP = False

error_order = 2

def set_order(order):
    global error_order
    error_order = order

def ufloat(mean, stddev):
    if stddev == 0:
        return float(mean)
    if error_order == 2 and _HAS_SOERP:
        return so.N(mean, stddev)
    return float(mean)

def value(x):
    if isinstance(x, float):
        return x
    if error_order == 2 and _HAS_SOERP:
        return x.mean
    return float(x)

def stddev(x):
    if isinstance(x, float):
        return 0
    if error_order == 2 and _HAS_SOERP:
        return math.sqrt(x.var)
    return 0

def log(x):
    if isinstance(x , float):
        return math.log(x)
    if error_order == 2 and _HAS_SOERP:
        return so_math.ln(x)
    return math.log(float(x))

def exp(x):
    if isinstance(x, float):
        return math.exp(x)
    if error_order == 2 and _HAS_SOERP:
        return so_math.exp(x)
    return math.exp(float(x))

def printVariable(name, x):
    print("(order " + str(error_order) + ") " + name + " = " + str(value(x)) + " +/- " + str(stddev(x)))
