from process import calculations


class MonteCarloRun:

    def __init__(self,
                 concordant_ages,
                 discordant_uPb,
                 discordant_pbPb,
                 pb_loss_ages,
                 reconstructed_ages_by_pb_loss_age,
                 statistics_by_pb_loss_age,
                 optimal_pb_loss_age):

        self.concordant_ages = concordant_ages
        self.concordant_uPb = [calculations.u238pb206_from_age(age) for age in concordant_ages]
        self.concordant_pbPb = [calculations.pb207pb206_from_age(age) for age in concordant_ages]

        self.discordant_uPb = discordant_uPb
        self.discordant_pbPb = discordant_pbPb

        self.pb_loss_ages = pb_loss_ages
        self.reconstructed_ages_by_pb_loss_age = reconstructed_ages_by_pb_loss_age
        self.statistics_by_pb_loss_age = statistics_by_pb_loss_age

        self.optimal_pb_loss_age = optimal_pb_loss_age
        self.optimal_uPb = calculations.u238pb206_from_age(optimal_pb_loss_age)
        self.optimal_pbPb = calculations.pb207pb206_from_age(optimal_pb_loss_age)
        self.optimal_statistic = statistics_by_pb_loss_age[optimal_pb_loss_age]
