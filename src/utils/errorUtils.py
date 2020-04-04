"""
Provides an abstraction over the different methods of error propagation during
mathematical calculations
"""

#import uncertainties as fo
#import uncertainties.umath as fo_math
import soerp as so
import soerp.umath as so_math
#import mcerp as mc
#import mcerp.umath as mc_math
import math

#mc.npts = config.MONTE_CARLO_SAMPLES
error_order = 2

def set_order(order):
    global error_order
    error_order = order

def ufloat(mean, stddev):
    if stddev == 0:
        return float(mean)
    if error_order == 2:
        return so.N(mean, stddev)
    """
    if error_order == 1:
        return fo.ufloat(mean, stddev)
    if error_order == "mc":
        return mc.N(mean, stddev)
    """

def value(x):
    if isinstance(x, float):
        return x
    if error_order == 2:
        return x.mean
    """
    if error_order == 1:
        return x.nominal_value
    if error_order == "mc":
        return x.mean
    """

def stddev(x):
    if isinstance(x, float):
        return 0
    if error_order == 2:
        return math.sqrt(x.var)
    """
    if error_order == 1:
        return x.std_dev
    if error_order == "mc":
        return math.sqrt(x.var)
    """

def log(x):
    if isinstance(x , float):
        return math.log(x)
    if error_order == 2:
        return so_math.ln(x)
    """
    if error_order == 1:
        return fo_math.log(x)
    if error_order == "mc":
        return mc_math.ln(x)
    """

def exp(x):
    if isinstance(x, float):
        return math.exp(x)
    if error_order == 2:
        return so_math.exp(x)
    """
    if error_order == 1:
        return fo_math.exp(x)
    if error_order == "mc":
        return mc_math.exp(x)
    """

def printVariable(name, x):
    print("(order " + str(error_order) + ") " + name + " = " + str(value(x)) + " +/- " + str(stddev(x)))