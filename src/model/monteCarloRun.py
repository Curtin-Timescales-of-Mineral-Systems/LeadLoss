from process import calculations, processing
import numpy as np
from scipy.stats import stats


class MonteCarloRunPbLossAgeStatistics:
    def __init__(self, concordant_ages, discordant_ages, dissimilarity_test, penalise_invalid_ages):
        self.valid_concordant_ages = [age for age in concordant_ages if age]
        self.valid_discordant_ages = [age for age in discordant_ages if age]

        self.number_of_ages = len(discordant_ages)
        self.number_of_invalid_ages = len(discordant_ages) - len(self.valid_discordant_ages)
        self.test_statistics = dissimilarity_test.perform(self.valid_concordant_ages, self.valid_discordant_ages)
        self.test_score = dissimilarity_test.getComparisonValue(self.test_statistics)

        if penalise_invalid_ages:
            # Lower scores are better
            invalid_fraction = 1 - len(self.valid_discordant_ages) / len(discordant_ages)
            self.score = self.test_score + (1 - self.test_score) * invalid_fraction
        else:
            self.score = self.test_score

class MonteCarloRun:
    def __init__(self, 
                 run_number,
                 sample_name,
                 concordant_uPb,
                 concordant_pbPb,
                 discordant_uPb,
                 discordant_pbPb):

        self.run_number = run_number
        self.sample_name = sample_name
        self.concordant_uPb = concordant_uPb
        self.concordant_pbPb = concordant_pbPb
        self.discordant_uPb = discordant_uPb
        self.discordant_pbPb = discordant_pbPb
        

        self.concordant_ages = []
        for concordantUPb, concordantPbPb in zip(concordant_uPb, concordant_pbPb):
            concordantAge = calculations.concordant_age(concordantUPb, concordantPbPb)
            self.concordant_ages.append(concordantAge)

        self.statistics_by_pb_loss_age = {}

        self.optimal_pb_loss_age = None
        self.optimal_uPb = None
        self.optimal_pbPb = None
        self.optimal_statistic = None

        self.heatmapColumnData = None

        self.lead_loss_ages = []

    def samplePbLossAge(self, leadLossAge, dissimilarity_test, penalise_invalid_ages):
        leadLossUPb = calculations.u238pb206_from_age(leadLossAge)
        leadLossPbPb = calculations.pb207pb206_from_age(leadLossAge)

        concordant_ages = self.concordant_ages
        discordant_ages = []
        for discordantUPb, discordantPbPb in zip(self.discordant_uPb, self.discordant_pbPb):
            discordant_age = calculations.discordant_age(leadLossUPb, leadLossPbPb, discordantUPb, discordantPbPb)
            discordant_ages.append(discordant_age)

        self.statistics_by_pb_loss_age[leadLossAge] = MonteCarloRunPbLossAgeStatistics(
            concordant_ages,
            discordant_ages,
            dissimilarity_test,
            penalise_invalid_ages
        )

        self.lead_loss_ages.append(leadLossAge)

    def calculateOptimalAge(self):
        results = [(age, statistic.score) for age, statistic in self.statistics_by_pb_loss_age.items()]
        results.sort(key=lambda v: v[0])
        valuesToCompare = [v for k, v in results]

        optimalLeadLossAgeIndex = processing._findOptimalIndex(valuesToCompare)
        optimalLeadLossAge = results[optimalLeadLossAgeIndex][0]

        self.optimal_pb_loss_age = optimalLeadLossAge
        self.optimal_uPb = calculations.u238pb206_from_age(optimalLeadLossAge)
        self.optimal_pbPb = calculations.pb207pb206_from_age(optimalLeadLossAge)
        self.optimal_statistic = self.statistics_by_pb_loss_age[optimalLeadLossAge]

    def createHeatmapData(self, minAge, maxAge, resolution):
        ageInc = (maxAge - minAge) / resolution
        runAges = sorted(list(self.statistics_by_pb_loss_age.keys()))
        colAges = [[] for _ in range(resolution)]
        for age in runAges:
            if age == maxAge:
                col = resolution - 1
            else:
                col = int((age - minAge) // ageInc)
            colAges[col].append(age)

        colData = []
        for col in range(resolution):
            prevNonEmptyCol = col
            nextNonEmptyCol = col
            while prevNonEmptyCol > 0 and len(colAges[prevNonEmptyCol]) == 0:
                prevNonEmptyCol -= 1
            while nextNonEmptyCol < resolution - 1 and len(colAges[nextNonEmptyCol]) == 0:
                nextNonEmptyCol += 1

            if len(colAges[prevNonEmptyCol]) == 0 or len(colAges[nextNonEmptyCol]) == 0:
                continue

            if prevNonEmptyCol != nextNonEmptyCol:
                prevAge = max(colAges[prevNonEmptyCol])
                nextAge = min(colAges[nextNonEmptyCol])
                prevStat = self.statistics_by_pb_loss_age[prevAge].test_statistics[0]
                nextStat = self.statistics_by_pb_loss_age[nextAge].test_statistics[0]
                prevDiff = col - prevNonEmptyCol
                nextDiff = nextNonEmptyCol - col
                totalDiff = nextDiff + prevDiff
                value = (nextDiff * prevStat + prevDiff * nextStat) / totalDiff
            else:
                value = np.mean([self.statistics_by_pb_loss_age[age].score for age in colAges[col]])
            colData.append(value)
        self.heatmapColumnData = colData

    def toList(self):
        return [self.sample_name, self.run_number, self.optimal_pb_loss_age / 1_000_000] #Convert to Ma