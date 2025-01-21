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
LOWER_AGE = 1

################
## Geological ##
################

def age_from_206Pb238U(ratio):
    return math.log(ratio+1) / U238_DECAY_CONSTANT

def age_from_207Pb235(ratio):
    return math.log(ratio+1) / U235_DECAY_CONSTANT

def pb206u238_from_age(age):
    return errors.exp(U238_DECAY_CONSTANT * age) - 1

def pb207u235_from_age(age):
    return errors.exp(U235_DECAY_CONSTANT * age) - 1


########################
## Wetherill Distances ##
########################

def discordance_wetherill(r207_235, r206_238):

    t207 = age_from_207Pb235(r207_235)
    t206 = age_from_206Pb238U(r206_238)
    result = (t207 - t206) / t207
    # Get rid of floating point inaccuracies
    if result > 10 ** -10:
        return result
    return 0.0

def concordant_age_wetherill(r206_238, r207_235):
    """
    Minimises distance in Wetherill coords:
      X(t)=207Pb/235U(t), Y(t)=206Pb/238U(t)
    from the measured (r207_235, r206_238).
    """
    def distance(t):
        x_theory = pb207u235_from_age(t)
        y_theory = pb206u238_from_age(t)
        dx = r207_235 - x_theory
        dy = r206_238 - y_theory
        return math.hypot(dx, dy)

    result = minimize_scalar(distance, method='Bounded',
                          bounds=[LOWER_AGE, UPPER_AGE])
    return result.x

def discordant_age_wetherill(x1, y1, x2, y2):
    if x1 >= x2:
        return None

    anchor_age = age_from_207Pb235(x1)
    # Shift bracket so we skip the anchor root
    small_margin = 1e3 # 1000 years
    bracket_start = anchor_age + small_margin
    if bracket_start >= UPPER_AGE:
        return None

    m = (y2 - y1)/(x2 - x1)
    c = y1 - m*x1

    # lower_limit = age_from_207Pb235(min(errors.value(x1), errors.value(x2)))
    # upper_limit = UPPER_AGE

    def func(t):
        x_t = pb207u235_from_age(t)
        y_t = pb206u238_from_age(t)
        line_y = m*x_t + c
        return y_t - line_y 

    v1 = func(bracket_start)
    v2 = func(UPPER_AGE)


    # v1 = func(lower_limit)
    # v2 = func(upper_limit)
    # print("DEBUG: bracket in Wetherill chord solver is", lower_limit, UPPER_AGE)
    if (v1 > 0 and v2 > 0) or (v1 < 0 and v2 < 0):
        # print("DEBUG => no sign change, skipping solver. v1=", v1, "v2=", v2)
        return None

    result = root_scalar(func, bracket=(bracket_start, UPPER_AGE))
    if not result.converged:
        return None
    return result.root

######################
## Covariance, etc. ##
######################

def mahalanobisRadius(sigmas):
    if sigmas==1:
        p=0.6827
    elif sigmas==2:
        p=0.9545
    else:
        raise Exception("Cannot handle # sigmas= "+str(sigmas))
    return -2.* math.log(1-p)

def isConcordantErrorEllipseWetherill(
    ratio207_235, error207_235,
    ratio206_238, error206_238,
    ellipseSigmas
):
    """
    Wetherill coords X=207/235, Y=206/238
    """
    s = mahalanobisRadius(ellipseSigmas)

    # check degenerate
    if error206_238== 0:
        try:
            root_s= math.sqrt(s)
            t_guess= age_from_206Pb238U(ratio206_238)
            x_theory= pb207u235_from_age(t_guess)
            dx= abs(ratio207_235- x_theory)
            return (dx <= error207_235* root_s)
        except:
            return False
    if error207_235==0.:
        try:
            root_s= math.sqrt(s)
            t_guess= age_from_207Pb235(ratio207_235)
            y_theory= pb206u238_from_age(t_guess)
            dy= abs(ratio206_238- y_theory)
            return (dy<= error206_238* root_s)
        except:
            return False

    # Minimization approach
    def distanceToEllipse(t):
        x_th= pb207u235_from_age(t)
        y_th= pb206u238_from_age(t)
        dx= (ratio207_235- x_th)/ error207_235
        dy= (ratio206_238- y_th)/ error206_238
        return dx*dx+dy*dy

    result = minimize_scalar(distanceToEllipse, bracket=(1,5.*(10**9)))
    if not result.success:
        raise Exception("Error minimising wetherill ellipse:\n"+ result.message)
    return (result.fun<= s)

#############
## General ##
#############

def to1StdDev(value, error, form, sigmas):
    if form=="Percentage":
        error= (error/100.)* value
    return error/ sigmas

def from1StdDev(value, error, form, sigmas):
    e= error* sigmas
    if form=="Percentage":
        return 100.* e/ value
    return e

def convert_from_stddev_without_sigmas(value, error, form):
    if form=="Percentage":
        return 100.* error/ value
    return error