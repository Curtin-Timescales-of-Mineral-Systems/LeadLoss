
import math

def ellipseAxes(xErr, xErrSigmas, yErr, yErrSigmas, ellipseSigmas):
    xErrSigma = xErr/xErrSigmas
    yErrSigma = yErr/yErrSigmas
    return xErrSigma * root_s, yErrSigma * root_s