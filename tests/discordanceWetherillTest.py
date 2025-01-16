import math

U238_DECAY_CONSTANT = 1.55125*(10**-10)
U235_DECAY_CONSTANT = 9.8485*(10**-10)
U238U235_RATIO = 137.818

def discordance_wetherill(r206_238, r207_235):
    t1 = age_from_206Pb238U(r206_238)
    t2 = age_from_207Pb235(r207_235)
    diff = (t2 - t1)/t2
    return abs(diff)

def age_from_206Pb238U(ratio):
    return math.log(ratio + 1) / U238_DECAY_CONSTANT

def age_from_207Pb235(ratio):
    return math.log(ratio + 1) / U235_DECAY_CONSTANT

def is_discordant(diff, cutoff = 0.10):
    """ Return true if 'diff' (in decimal fraction)is >= cutoff, else False """
    return (diff >= cutoff)

def main():
    data = [
    (0.1680, 0.0020, 1.680, 0.030),
    (0.1680, 0.0020, 2.100, 0.030),
    (0.3640, 0.0050, 6.160, 0.080),
    (0.5930, 0.0080, 18.20, 0.30),
    (0.8000, 0.0200, 2.000, 0.050),
    ]

    # Test only percenrage based discordance approach here
    # so we do: disc = discordance_wetherill(r206_238, r207_235)
    # Then if disc >= 0.10, 'discordant' else 'concordant'
    cutoff_fraction = 0.10

    for i, (r206_238, err206_238, r207_235, err207_235) in enumerate(data):
        disc = discordance_wetherill(r206_238, r207_235)
        # disc is a decimal fraction, e.g. 0.12 means 12% difference
        # or 0.02 means 2% difference
        # We'll classify:
        is_conc = not is_discordant(disc, cutoff_fraction)
        # Print a quick summary
        label = f"Row{i+1}"
        print(f"{label}: 206/238={r206_238}, 207/235={r207_235}, disc={disc*100:.2f}% => ", end="")
        print("Concordant" if is_conc else "Discordant")

if __name__ == "__main__":
    main()