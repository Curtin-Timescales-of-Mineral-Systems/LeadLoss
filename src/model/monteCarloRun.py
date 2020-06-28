from process import calculations, processing
import numpy as np
from scipy.stats import stats

class MonteCarloRun:

    def __init__(self,
                 concordant_uPb,
                 concordant_pbPb,
                 discordant_uPb,
                 discordant_pbPb):

        self.concordant_uPb = concordant_uPb
        self.concordant_pbPb = concordant_pbPb
        self.discordant_uPb = discordant_uPb
        self.discordant_pbPb = discordant_pbPb
        self.concordant_ages = []
        for concordantUPb, concordantPbPb in zip(concordant_uPb, concordant_pbPb):
            concordantAge = calculations.concordant_age(concordantUPb, concordantPbPb)
            self.concordant_ages.append(concordantAge)

        self.pb_loss_ages = []
        self.reconstructed_ages_by_pb_loss_age = {}
        self.statistics_by_pb_loss_age = {}

        self.optimal_pb_loss_age = None
        self.optimal_uPb = None
        self.optimal_pbPb = None
        self.optimal_statistic = None

        self.heatmapColumnData = None

    def samplePbLossAge(self, leadLossAge):
        leadLossUPb = calculations.u238pb206_from_age(leadLossAge)
        leadLossPbPb = calculations.pb207pb206_from_age(leadLossAge)

        concordantAges = self.concordant_ages
        discordantAges = []
        for discordantUPb, discordantPbPb in zip(self.discordant_uPb, self.discordant_pbPb):
            discordantAge = calculations.discordant_age(leadLossUPb, leadLossPbPb, discordantUPb, discordantPbPb)
            discordantAges.append(discordantAge)

        if not discordantAges or not concordantAges:
            statistics = (1.0, 0.0)
        else:
            distribution1 = [age if age else 0 for age in concordantAges]
            distribution2 = [age if age else 0 for age in discordantAges]
            statistics = stats.ks_2samp(distribution1, distribution2)

        self.pb_loss_ages.append(leadLossAge)
        self.reconstructed_ages_by_pb_loss_age[leadLossAge] = discordantAges
        self.statistics_by_pb_loss_age[leadLossAge] = statistics


    def calculateOptimalAge(self, dissimilarityTest):
        results = [(age, dissimilarityTest.getComparisonValue(statistics)) for (age, statistics) in self.statistics_by_pb_loss_age.items()]
        results.sort(key=lambda v: v[0])
        valuesToCompare = [v for k,v in results]

        optimalLeadLossAgeIndex = processing._findOptimalIndex(valuesToCompare)
        optimalLeadLossAge = self.pb_loss_ages[optimalLeadLossAgeIndex]

        self.optimal_pb_loss_age = optimalLeadLossAge
        self.optimal_uPb = calculations.u238pb206_from_age(optimalLeadLossAge)
        self.optimal_pbPb = calculations.pb207pb206_from_age(optimalLeadLossAge)
        self.optimal_statistic = self.statistics_by_pb_loss_age[optimalLeadLossAge]

    def createHeatmapData(self, minAge, maxAge, resolution):
        ageInc = (maxAge - minAge) / resolution
        runAges = sorted(self.pb_loss_ages)
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
                prevStat = self.statistics_by_pb_loss_age[prevAge][0]
                nextStat = self.statistics_by_pb_loss_age[nextAge][0]
                prevDiff = col - prevNonEmptyCol
                nextDiff = nextNonEmptyCol - col
                totalDiff = nextDiff + prevDiff
                value = (nextDiff * prevStat + prevDiff * nextStat) / totalDiff
            else:
                value = np.mean([self.statistics_by_pb_loss_age[age][0] for age in colAges[col]])
            colData.append(value)
        self.heatmapColumnData = colData

