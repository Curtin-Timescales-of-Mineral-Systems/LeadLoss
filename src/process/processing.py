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
    concordancy = []
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

        discordances.append(discordance)
        concordancy.append(concordant)

    sample.updateConcordance(concordancy, discordances)
    signals.progress(ProgressType.CONCORDANCE, 1.0, sample.name, concordancy, discordances)

    return True


def _performRimAgeSampling(signals, sample):
    sampleNameText = " for '" + sample.name + "'" if sample.name else ""
    signals.newTask("Sampling Pb-loss age distributions" + sampleNameText + "...")
    # Actually compute the age distributions and statistics
    settings = sample.calculationSettings

    # Get the concordant samples
    concordantSpots = [spot for spot in sample.validSpots if spot.concordant]
    discordantSpots = [spot for spot in sample.validSpots if not spot.concordant]

    # Generate the discordant samples
    stabilitySamples = settings.monteCarloRuns

    # Reseed to avoid generating the same random numbers every time
    random = np.random.RandomState()

    concordantUPbValues = np.transpose(
        [random.normal(row.uPbValue, row.uPbStDev, stabilitySamples) for row in concordantSpots])
    concordantPbPbValues = np.transpose(
        [random.normal(row.pbPbValue, row.pbPbStDev, stabilitySamples) for row in concordantSpots])

    discordantUPbValues = np.transpose(
        [random.normal(row.uPbValue, row.uPbStDev, stabilitySamples) for row in discordantSpots])
    discordantPbPbValues = np.transpose(
        [random.normal(row.pbPbValue, row.pbPbStDev, stabilitySamples) for row in discordantSpots])

    for j in range(stabilitySamples):
        if signals.halt():
            signals.cancelled()
            return False

        runConcordantUPbValues = concordantUPbValues[j]
        runConcordantPbPbValues = concordantPbPbValues[j]
        runDiscordantUPbValues = discordantUPbValues[j]
        runDiscordantPbPbValues = discordantPbPbValues[j]

        run = _performSingleRun(settings, runConcordantUPbValues, runConcordantPbPbValues, runDiscordantUPbValues, runDiscordantPbPbValues)
        sample.addMonteCarloRun(run)

        progress = (j + 1) / stabilitySamples
        signals.progress(ProgressType.SAMPLING, progress, sample.name, run)
    return True

def _performSingleRun(settings, concordantUPbs, concordantPbPbs, discordantUPbs, discordantPbPbs):
    # Calculate concordant ages
    concordantAges = []
    for concordantUPb, concordantPbPb in zip(concordantUPbs, concordantPbPbs):
        concordantAge = calculations.concordant_age(concordantUPb, concordantPbPb)
        concordantAges.append(concordantAge)

    # Generate the lead loss age samples
    leadLossAges = settings.rimAges()
    runStatistics = []
    runReconstructedAgesByLeadLossAge = {}
    for leadLossAge in leadLossAges:
        discordantAges, statistics = _performSingleLeadLossAgeSample(concordantAges, leadLossAge, discordantUPbs, discordantPbPbs)
        runReconstructedAgesByLeadLossAge[leadLossAge] = discordantAges
        runStatistics.append(statistics)

    valuesToCompare = [settings.dissimilarityTest.getComparisonValue(s) for s in runStatistics]
    optimalLeadLossAgeIndex = _findOptimalIndex(valuesToCompare)
    optimalLeadLossAge = leadLossAges[optimalLeadLossAgeIndex]
    runStatisticsByLeadLossAge = {age: stat for age, stat in zip(leadLossAges, runStatistics)}

    return MonteCarloRun(
        concordantAges, concordantUPbs, concordantPbPbs,
        discordantUPbs, discordantPbPbs,
        leadLossAges,
        runReconstructedAgesByLeadLossAge,
        runStatisticsByLeadLossAge,
        optimalLeadLossAge
    )

def _performSingleLeadLossAgeSample(concordantAges, leadLossAge, discordantUPbs, discordantPbPbs):
    leadLossUPb = calculations.u238pb206_from_age(leadLossAge)
    leadLossPbPb = calculations.pb207pb206_from_age(leadLossAge)

    discordantAges = []
    for discordantUPb, discordantPbPb in zip(discordantUPbs, discordantPbPbs):
        discordantAge = calculations.discordant_age(leadLossUPb, leadLossPbPb, discordantUPb, discordantPbPb)
        discordantAges.append(discordantAge)

    statistics = _calculateStatistics(concordantAges, discordantAges)

    return discordantAges, statistics

def _calculateStatistics(concordantAges, reconstructedAges):
    if not reconstructedAges or not concordantAges:
        return 0

    distribution1 = [age if age else 0 for age in concordantAges]
    distribution2 = [age if age else 0 for age in reconstructedAges]
    return stats.ks_2samp(distribution1, distribution2)


def _findOptimalIndex(valuesToCompare):
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
    ages = settings.rimAges()
    values = [np.mean([run.statistics_by_pb_loss_age[age][0] for run in sample.monteCarloRuns]) for age in ages]
    optimalAgeIndex = _findOptimalIndex(values)
    optimalAge = ages[optimalAgeIndex]

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
