import math
from enum import Enum

from scipy.stats import stats
import numpy as np

from model.monteCarloRun import MonteCarloRun
from src.model.settings.calculation import DiscordanceClassificationMethod
from process import calculations
import time

TIME_PER_TASK = 0.0


class ProgressType(Enum):
    CONCORDANCE = 0,
    SAMPLING = 1,
    OPTIMAL = 2,


def processSamples(signals, samples):
    for sample in samples:
        completed = _processSample(signals, sample)
        if not completed:
            return

    signals.completed()

def _processSample(signals, sample):
    completed = _calculateConcordantAges(signals, sample)
    if not completed:
        return False

    completed = _performRimAgeSampling(signals, sample)
    if not completed:
        return False

    completed = _calculateOptimalAge(signals, sample)
    if not completed:
        return False

    return True

def _calculateConcordantAges(signals, sample):
    sampleNameText = " for '" + sample.name + "'" if sample.name else ""
    signals.newTask("Classifying points" + sampleNameText + "...")

    settings = sample.calculationSettings
    timePerRow = TIME_PER_TASK / len(sample.validSpots)
    concordantAges = []
    discordances = []
    for i, spot in enumerate(sample.validSpots):
        signals.progress(ProgressType.CONCORDANCE, i / len(sample.validSpots))

        time.sleep(timePerRow)
        if signals.halt():
            signals.cancelled()
            return False

        if settings.discordanceClassificationMethod == DiscordanceClassificationMethod.PERCENTAGE:
            discordance = calculations.discordance(spot.uPbValue, spot.pbPbValue)
            concordant = discordance < settings.discordancePercentageCutoff
        else:
            discordance = None
            concordant = calculations.isConcordantErrorEllipse(
                spot.uPbValue,
                spot.uPbStDev,
                spot.pbPbValue,
                spot.pbPbStDev,
                settings.discordanceEllipseSigmas
            )

        if concordant:
            concordantAge = calculations.concordant_age(spot.uPbValue, spot.pbPbValue)
        else:
            concordantAge = None

        discordances.append(discordance)
        concordantAges.append(concordantAge)

    sample.updateConcordance(concordantAges, discordances)
    signals.progress(ProgressType.CONCORDANCE, 1.0, sample.name, concordantAges, discordances)

    return True


def _performRimAgeSampling(signals, sample):
    sampleNameText = " for '" + sample.name + "'" if sample.name else ""
    signals.newTask("Sampling Pb-loss age distributions" + sampleNameText + "...")
    # Actually compute the age distributions and statistics
    settings = sample.calculationSettings
    dissimilarityTest = settings.dissimilarityTest

    # Get the concordant samples
    concordantAges = [spot.concordantAge for spot in sample.validSpots if spot.concordant]
    discordantSpots = [spot for spot in sample.validSpots if not spot.concordant]

    # Generate the discordant samples
    stabilitySamples = settings.monteCarloRuns
    discordantUPbValues = np.transpose(
        [np.random.normal(row.uPbValue, row.uPbStDev, stabilitySamples) for row in discordantSpots])
    discordantPbPbValues = np.transpose(
        [np.random.normal(row.pbPbValue, row.pbPbStDev, stabilitySamples) for row in discordantSpots])

    # Generate the lead loss age samples
    leadLossAgeSamples = settings.rimAgesSampled
    leadLossAges = settings.rimAges()
    leadLossUPbValues = [calculations.u238pb206_from_age(age) for age in leadLossAges]
    leadLossPbPbValues = [calculations.pb207pb206_from_age(age) for age in leadLossAges]

    for j in range(stabilitySamples):

        runDiscordantUPbValues = discordantUPbValues[j]
        runDiscordantPbPbValues = discordantPbPbValues[j]
        runStatistics = []
        runReconstructedAgesByLeadLossAge = {}
        for i in range(leadLossAgeSamples):
            if signals.halt():
                signals.cancelled()
                return False

            leadLossAge = leadLossAges[i]
            leadLossUPb = leadLossUPbValues[i]
            leadLossPbPb = leadLossPbPbValues[i]

            reconstructedAges = []
            for k, spot in enumerate(discordantSpots):
                discordantUPb = runDiscordantUPbValues[k]
                discordantPbPb = runDiscordantPbPbValues[k]
                reconstructedAge = calculations.discordant_age(leadLossUPb, leadLossPbPb, discordantUPb, discordantPbPb)
                reconstructedAges.append(reconstructedAge)

            runReconstructedAgesByLeadLossAge[leadLossAge] = reconstructedAges
            runStatistics.append(_calculateStatistics(concordantAges, reconstructedAges))

        optimalLeadLossAgeIndex = _findOptimalAgeIndex(dissimilarityTest, runStatistics)
        optimalLeadLossAge = leadLossAges[optimalLeadLossAgeIndex]
        runStatisticsByLeadLossAge = {age: stat for age, stat in zip(leadLossAges, runStatistics)}

        run = MonteCarloRun(concordantAges,
                            runDiscordantUPbValues,
                            runDiscordantPbPbValues,
                            leadLossAges,
                            runReconstructedAgesByLeadLossAge,
                            runStatisticsByLeadLossAge,
                            optimalLeadLossAge)

        sample.addMonteCarloRun(run)

        progress = (j + 1) / stabilitySamples
        signals.progress(ProgressType.SAMPLING, progress, sample.name, run)
    return True

def _calculateStatistics(concordantAges, reconstructedAges):
    if not reconstructedAges or not concordantAges:
        return 0

    distribution1 = [age if age else 0 for age in concordantAges]
    distribution2 = [age if age else 0 for age in reconstructedAges]
    return stats.ks_2samp(distribution1, distribution2)


def _findOptimalAgeIndex(dissimilarityTest, runStatistics):
    valuesToCompare = [dissimilarityTest.getComparisonValue(s) for s in runStatistics]
    minIndex, minValue = min(enumerate(valuesToCompare), key=lambda v: v[1])
    n = len(valuesToCompare)

    startMinIndex = minIndex
    while startMinIndex > 0 and valuesToCompare[startMinIndex - 1] == minValue:
        startMinIndex -= 1

    endMinIndex = minIndex
    while endMinIndex < n - 1 and valuesToCompare[endMinIndex + 1] == minValue:
        endMinIndex += 1

    if  (endMinIndex != n - 1 and startMinIndex != 0) or \
        (endMinIndex == n - 1 and startMinIndex == 0):
        return (endMinIndex + startMinIndex) // 2

    if startMinIndex == 0:
        return 0

    return n - 1

def _calculateOptimalAge(signals, sample):
    settings = sample.calculationSettings

    # Find optimal age
    optimalStatistic = float('inf')
    optimalAge = None
    for age in settings.rimAges():
        value = np.mean([run.statistics_by_pb_loss_age[age][0] for run in sample.monteCarloRuns])
        if value <= optimalStatistic:
            optimalAge = age
            optimalStatistic = value

    # Find 95% confidence interval around optimal age
    optimalAges = [run.optimal_pb_loss_age for run in sample.monteCarloRuns]
    optimalAges.sort()
    n = len(optimalAges)
    cutoff2p5 = int(math.floor(0.025*n))
    cutoff97p5 = int(math.ceil(0.975*n)) - 1
    optimalAgeLowerBound = optimalAges[cutoff2p5]
    optimalAgeUpperBound = optimalAges[cutoff97p5]

    # Find mean D-value and p-value for optimal age
    optimalMeanDValue = np.mean([run.statistics_by_pb_loss_age[optimalAge][0] for run in sample.monteCarloRuns])
    optimalMeanPValue = np.mean([run.statistics_by_pb_loss_age[optimalAge][1] for run in sample.monteCarloRuns])

    # Return results
    args = optimalAge, optimalAgeLowerBound, optimalAgeUpperBound, optimalMeanDValue, optimalMeanPValue
    signals.progress(ProgressType.OPTIMAL, 1.0, sample.name, args)
    return True
