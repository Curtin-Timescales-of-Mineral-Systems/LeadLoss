
import math

def ellipseAxes(xErr, xErrSigmas, yErr, yErrSigmas, ellipseSigmas):
    xErrSigma = xErr/xErrSigmas
    yErrSigma = yErr/yErrSigmas
    root_s = math.sqrt(float(ellipseSigmas))
    return xErrSigma * root_s, yErrSigma * root_s
