import math
import random

from process import calculations


def generate():
    print("U238/Pb206, error, Pb207/Pb206, error")
    points = 20
    rimAge = 1000*(10**6)
    rimUPb = calculations.u238pb206_from_age(rimAge)
    rimPbPb = calculations.pb207pb206_from_age(rimAge)

    for i in range(points):
        reconstructedAge = random.randrange(2500*(10**6), 4500*(10**6))
        reconstructedUPb = calculations.u238pb206_from_age(reconstructedAge)
        reconstructedPbPb = calculations.pb207pb206_from_age(reconstructedAge)
        reconstructedUPbError = random.random()*0.1
        reconstructedPbPbError = random.random()*0.05
        print(",".join([str(reconstructedUPb), str(reconstructedUPbError), str(reconstructedPbPb), str(reconstructedPbPbError)]))

        fraction = random.uniform(0.2,0.8)
        discordantUPb = reconstructedUPb + (rimUPb - reconstructedUPb)*fraction
        discordantPbPb = reconstructedPbPb + (rimPbPb - reconstructedPbPb)*fraction
        discordantUPbError = random.random()*0.1
        discordantPbPbError = random.random()*0.05
        print(",".join([str(discordantUPb), str(discordantUPbError), str(discordantPbPb), str(discordantPbPbError)]))

if __name__ == "__main__":
    generate()