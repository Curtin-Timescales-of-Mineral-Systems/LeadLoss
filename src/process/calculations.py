import math

from scipy.optimize import root_scalar, minimize_scalar

import utils.errorUtils as errors

###############
## Constants ##
###############

from process.reconstructedAge import ReconstructedAge

U238_DECAY_CONSTANT = 1.55125*(10**-10)
U235_DECAY_CONSTANT = 9.8485*(10**-10)
U238U235_RATIO = 137.818

UPPER_AGE = 6000 * (10 ** 6)
LOWER_AGE = 1 * (10 ** 6)

################
## Geological ##
################

def age_from_u238pb206(u238pb206):
    return errors.log(1 / u238pb206 + 1) / U238_DECAY_CONSTANT

def age_from_pb207pb206(pb207pb206):
    return root_scalar(lambda t : pb207pb206_from_age(t) - pb207pb206, x0=9**10, bracket=[1, 10**10]).root

def pb206u238_from_age(age):
    return errors.exp(U238_DECAY_CONSTANT * age) - 1

def u238pb206_from_age(age):
    return 1/(pb206u238_from_age(age))

def pb207u235_from_age(age):
    return errors.exp(U235_DECAY_CONSTANT * age) - 1

def pb207pb206_from_age(age):
    pb207u235 = pb207u235_from_age(age)
    u238pb206 = u238pb206_from_age(age)
    return pb207u235*(1/U238U235_RATIO)*u238pb206

def pb207pb206_from_u238pb206(u238pb206):
    age = age_from_u238pb206(u238pb206)
    return pb207pb206_from_age(age)

def u238pb206_from_pb207pb206(pb207pb206):
    age = age_from_pb207pb206(pb207pb206)
    return u238pb206_from_age(age)

def discordance(u238pb206, pb207pb206):
    uPbAge = age_from_u238pb206(u238pb206)
    pbPbAge = age_from_pb207pb206(pb207pb206)
    result = (pbPbAge - uPbAge) / pbPbAge

    # Get rid of floating point inaccuracies
    if result > 10 ** -10:
        return result
    return 0.0

def concordant_age(u238pb206, pb207pb206):
    def distance(t):
        x = u238pb206 - u238pb206_from_age(t)
        y = pb207pb206 - pb207pb206_from_age(t)
        d = math.hypot(x, y)
        return d
    result = minimize_scalar(distance, method='Bounded', bounds=[LOWER_AGE, UPPER_AGE])
    return result.x

def discordant_age(x1, y1, x2, y2):
    if x1 <= x2:
        return None

    m = (y2 - y1)/(x2 - x1)
    c = y1 - m*x1

    lower_limit = age_from_u238pb206(min(errors.value(x1), errors.value(x2)))
    upper_limit = UPPER_AGE

    def func(t):
        curve_pb207pb206_value = pb207pb206_from_age(t)
        line_pb207pb206 = m*u238pb206_from_age(t) + c
        line_pb207pb206_value = line_pb207pb206
        return curve_pb207pb206_value - line_pb207pb206_value

    v1 = func(lower_limit)
    v2 = func(upper_limit)

    if (v1 > 0 and v2 > 0) or (v1 < 0 and v2 < 0):
        return None

    result = root_scalar(func, bracket=(lower_limit, upper_limit))
    return result.root


def mahalanobisRadius(sigmas):
    if sigmas == 1:
        p = 0.6827
    elif sigmas == 2:
        p = 0.9545
    else:
        raise Exception("Unable to handle " + str(sigmas) + " sigmas")
    return -2 * math.log(1 - p)

def isConcordantErrorEllipse(uPbValue, uPbError, pbPbValue, pbPbError, ellipseSigmas):
    """
    See https://www.xarg.org/2018/04/how-to-plot-a-covariance-error-ellipse/

    Keyword arguments:
    uPbValue -- the coordinate in TW concordia space
    uPbError -- the error associated with the U/Pb value (1 standard deviation)
    pbPbValue -- the x coordinate in TW concordia space
    pbPbError -- the error associated with the Pb/Pb value (1 standard deviation)
    """
    s = mahalanobisRadius(ellipseSigmas)

    # Handle degenerate cases
    if uPbError == 0:
        localPbPb = pb207pb206_from_u238pb206(uPbValue)
        root_s = math.sqrt(s)
        return abs(localPbPb-pbPbValue) <= pbPbError*root_s
    if pbPbError == 0:
        localUPb = u238pb206_from_pb207pb206(pbPbValue)
        root_s = math.sqrt(s)
        return abs(localUPb-uPbValue) <= uPbError*root_s


    # Otherwise minimise for distance in elliptical space
    def distanceToEllipse(t):
        x = u238pb206_from_age(t)
        y = pb207pb206_from_age(t)
        value = ((uPbValue-x)/uPbError)**2 + ((pbPbValue-y)/pbPbError)**2
        return value

    result = minimize_scalar(distanceToEllipse, bracket=(1,5.*(10**9)))
    if not result.success:
        raise Exception("Exception occurred while minimising distance to error ellipse:\n\n" + result.message)
    return result.fun <= s

#############
## General ##
#############

def to1StdDev(value, error, form, sigmas):
    if form == "Percentage":
        error = (error/100.0) * value
    return error/sigmas

def from1StdDev(value, error, form, sigmas):
    res = error*sigmas
    if form == "Percentage":
        return 100.0*res/value
    return res

def convert_from_stddev_without_sigmas(value, error, form):
    if form == "Percentage":
        return 100.0*error/value
    return error