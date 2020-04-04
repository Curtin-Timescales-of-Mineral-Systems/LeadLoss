import utils.calculations as calculations


def getCSVLine(x, y):
    return str(x) + ",0," + str(y) + ",0"

def test():
    rimAge = 10 * (10 ** 8)
    actualAge = 35 * (10 ** 8)

    x1 = calculations.u238pb206_from_age(rimAge)
    x2 = calculations.u238pb206_from_age(actualAge)
    y1 = calculations.pb207pb206_from_age(rimAge)
    y2 = calculations.pb207pb206_from_age(actualAge)

    midAgeX = (x1 + x2)/2
    midAgeY = (y1 + y2)/2

    print(getCSVLine(x2, y2))
    print(getCSVLine(midAgeX, midAgeY))

test()