from process import processing, calculationsWetherill
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
            invalid_fraction = 1 - len(self.valid_discordant_ages) / len(discordant_ages)
            self.score = self.test_score + (1 - self.test_score) * invalid_fraction
        else:
            self.score = self.test_score


class MonteCarloRunWetherill:
    """
    Monte Carlo run in Wetherill space:
      X = 207Pb/235U
      Y = 206Pb/238U

    """

    def __init__(
        self,
        run_number,
        sample_name,
        concordant_206_238,
        concordant_207_235,
        discordant_206_238,
        discordant_207_235
    ):
        self.run_number = run_number
        self.sample_name = sample_name

        # Store Wetherill arrays
        self.concordant_206_238 = concordant_206_238
        self.concordant_207_235 = concordant_207_235
        self.discordant_206_238 = discordant_206_238
        self.discordant_207_235 = discordant_207_235

        # ALIASES for Tera–W naming => UI code referencing .concordant_uPb doesn’t break
        self.concordant_uPb = self.concordant_206_238
        self.concordant_pbPb = self.concordant_207_235
        self.discordant_uPb = self.discordant_206_238
        self.discordant_pbPb = self.discordant_207_235

        # Pre-compute minimal distances => "concordant_ages"
        self.concordant_ages = []
        for (r206_238, r207_235) in zip(concordant_206_238, concordant_207_235):
            age = calculationsWetherill.concordant_age_wetherill(r206_238, r207_235)
            self.concordant_ages.append(age)

        self.statistics_by_pb_loss_age = {}

        self.optimal_pb_loss_age = None
        self.optimal_206_238 = None
        self.optimal_207_235 = None
        self.optimal_statistic = None

        # ALIASES for old UI code
        self.optimal_uPb = None
        self.optimal_pbPb = None

        self.heatmapColumnData = None
        self.lead_loss_ages = []
        

    def samplePbLossAgeWetherill(
        self, leadLossAge, dissimilarity_test, penalise_invalid_ages
    ):
        """
        Wetherill version: 
          anchor in (X, Y) = (207Pb/235U, 206Pb/238U) at leadLossAge
          then compute 'discordant ages' via discordant_age_wetherill
        """
        x_anchor = calculationsWetherill.pb207u235_from_age(leadLossAge)
        y_anchor = calculationsWetherill.pb206u238_from_age(leadLossAge)

        concordant_ages = self.concordant_ages
        discordant_ages = []
        for (spot207_235, spot206_238) in zip(self.discordant_207_235, self.discordant_206_238):
            discordant_age = calculationsWetherill.discordant_age_wetherill(
                
                 x_anchor, y_anchor, spot207_235, spot206_238
            )
            discordant_ages.append(discordant_age)

        # print(f"DEBUG => For Pb-loss age={leadLossAge:.3f} Ma, the reconstructed ages are: {discordant_ages}")
        
        self.statistics_by_pb_loss_age[leadLossAge] = MonteCarloRunPbLossAgeStatistics(
            concordant_ages,
            discordant_ages,
            dissimilarity_test,
            penalise_invalid_ages
        )
        
        self.lead_loss_ages.append(leadLossAge)

    def calculateOptimalAge(self):
        """
        Finds the age with the minimal 'score', sets .optimal_pb_loss_age, 
        plus sets Wetherill coords => .optimal_206_238, .optimal_207_235,
        and also old .optimal_uPb, .optimal_pbPb for UI compatibility.
        """
        results = [(age, st.score) for age, st in self.statistics_by_pb_loss_age.items()]
        results.sort(key=lambda x: x[0])  # sort by age
        valuesToCompare = [v for k, v in results]

        # print("DEBUG => For run #", self.run_number, "the tested Pb-loss ages [score]:")
        # for (pb_age, stat_score) in results:
        #     print(f"   Age={pb_age:.2f}, Score={stat_score:.4f}")

        optimalLeadLossAgeIndex = processing._findOptimalIndex(valuesToCompare)
        optimalLeadLossAge = results[optimalLeadLossAgeIndex][0]

        self.optimal_pb_loss_age = optimalLeadLossAge
        self.optimal_206_238 = calculationsWetherill.pb206u238_from_age(optimalLeadLossAge)
        self.optimal_207_235 = calculationsWetherill.pb207u235_from_age(optimalLeadLossAge)
        self.optimal_statistic = self.statistics_by_pb_loss_age[optimalLeadLossAge]

        # For old UI code referencing .optimal_uPb/.optimal_pbPb
        self.optimal_uPb = self.optimal_207_235
        self.optimal_pbPb = self.optimal_206_238
        # print(f"DEBUG => Chosen optimal Pb-loss age for run #{self.run_number} is {optimalLeadLossAge:.2f} Ma\n")
    
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