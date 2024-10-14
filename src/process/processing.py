import math
import time
from enum import Enum

import numpy as np
import scipy as sp
from scipy.stats import stats

from model.monteCarloRun import MonteCarloRun
from process import calculations
from model.settings.calculation import DiscordanceClassificationMethod
from utils import config

TIME_PER_TASK = 0.0


class ProgressType(Enum):
    CONCORDANCE = 0,
    SAMPLING = 1,
    OPTIMAL = 2,

def processSamples(signals, samples):
    for sample in samples:
        _processSample(signals, sample)
    signals.completed()


def _processSample(signals, sample):
    completed = _calculateConcordantAges(signals, sample)
    if not completed:
        return False

    _performRimAgeSampling(signals, sample)

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
    
    # if discordances == 0:
    #     signals.progress(ProgressType.OPTIMAL, progress, sample.name, None)
    #     return True


def _performRimAgeSampling(signals, sample):
    sampleNameText = " for '" + sample.name + "'" if sample.name else ""
    signals.newTask("Sampling Pb-loss age distributions" + sampleNameText + "...")
    # Actually compute the age distributions and statistics
    settings = sample.calculationSettings

    # Get the concordant samples
    concordantSpots = [spot for spot in sample.validSpots if spot.concordant]
    discordantSpots = [spot for spot in sample.validSpots if not spot.concordant]

    # If either list is empty, return immediately
    if not concordantSpots or not discordantSpots:
        return False

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

    # Check if the lists are empty before processing
    # Skip samples that have 2 or less discordant spots
    if len(discordantSpots) <= 2:
        return False
    
    if len(concordantUPbValues) == 0 or len(discordantUPbValues) == 0:
        return False
    
    for j in range(stabilitySamples):
        if signals.halt():
            signals.cancelled()
            return False

        runConcordantUPbValues = concordantUPbValues[j]
        runConcordantPbPbValues = concordantPbPbValues[j]
        runDiscordantUPbValues = discordantUPbValues[j]
        runDiscordantPbPbValues = discordantPbPbValues[j]

        run = MonteCarloRun(
            j,
            sample.name,
            runConcordantUPbValues,
            runConcordantPbPbValues,
            runDiscordantUPbValues,
            runDiscordantPbPbValues,
        )

        _performSingleRun(settings, run)
        sample.addMonteCarloRun(run)

        progress = (j + 1) / stabilitySamples
        signals.progress(ProgressType.SAMPLING, progress, sample.name, run)
        if j % 5 == 0 or j == stabilitySamples - 1:
            _calculateOptimalAge(signals, sample, progress)
    return True

def _performSingleRun(settings, run):
    # Generate the lead loss age samples
    for age in settings.rimAges():
        run.samplePbLossAge(age, settings.dissimilarityTest, settings.penaliseInvalidAges)
    run.calculateOptimalAge()
    run.createHeatmapData(settings.minimumRimAge, settings.maximumRimAge, config.HEATMAP_RESOLUTION)

def _findOptimalIndex(valuesToCompare):
    minIndex, minValue = min(enumerate(valuesToCompare), key=lambda v: v[1])
    n = len(valuesToCompare)

    startMinIndex = minIndex
    while startMinIndex > 0 and valuesToCompare[startMinIndex - 1] == minValue:
        startMinIndex -= 1

    endMinIndex = minIndex
    while endMinIndex < n - 1 and valuesToCompare[endMinIndex + 1] == minValue:
        endMinIndex += 1

    if (endMinIndex != n - 1 and startMinIndex != 0) or (endMinIndex == n - 1 and startMinIndex == 0):
        return (endMinIndex + startMinIndex) // 2

    if startMinIndex == 0:
        return 0

    return n - 1


def _calculateOptimalAge(signals, sample, progress):
    settings = sample.calculationSettings
    runs = sample.monteCarloRuns

    # Find optimal age
    ages = settings.rimAges()
    values = [np.mean([run.statistics_by_pb_loss_age[age].score for run in runs]) for age in ages]
    optimalAgeIndex = _findOptimalIndex(values)
    optimalAge = ages[optimalAgeIndex]

    # Find 95% confidence interval around optimal age
    optimalAges = [run.optimal_pb_loss_age for run in runs]
    optimalAges.sort()
    n = len(optimalAges)
    cutoff2p5 = int(math.floor(0.025 * n))
    cutoff97p5 = int(math.ceil(0.975 * n)) - 1
    optimalAgeLowerBound = optimalAges[cutoff2p5]
    optimalAgeUpperBound = optimalAges[cutoff97p5]

    # Find mean D-value and p-value for optimal age
    optimalMeanDValue = np.mean([run.optimal_statistic.test_statistics[0] for run in runs])
    optimalMeanPValue = np.mean([run.optimal_statistic.test_statistics[1] for run in runs])
    optimalMeanNumberOfInvalidPoints = np.mean([run.optimal_statistic.number_of_invalid_ages for run in runs])
    optimalMeanScore = np.mean([run.optimal_statistic.score for run in runs])

    # Return results
    args = optimalAge, optimalAgeLowerBound, optimalAgeUpperBound, optimalMeanDValue, optimalMeanPValue, optimalMeanNumberOfInvalidPoints, optimalMeanScore
    signals.progress(ProgressType.OPTIMAL, progress, sample.name, args)
    return True


def calculateHeatmapData(signals, runs, settings):
    resolution = config.HEATMAP_RESOLUTION

    colData = [[] for _ in range(resolution)]
    for run in runs:
        for col in range(resolution):
            colData[col].append(run.heatmapColumnData[col])

    cache = {}
    data = [[0 for _ in range(resolution)] for _ in range(resolution)]
    for col in range(resolution):
        if len(colData[col]) == 0:
            continue

        mean = np.mean(colData[col])
        stdDev = np.std(colData[col])
        if stdDev < 10 ** -7:
            stdDev = 0

        if (mean, stdDev) not in cache:
            if stdDev == 0:
                if mean == 1.0:
                    meanRow = resolution - 1
                else:
                    meanRow = int(mean * resolution)
                result = [1 if i >= meanRow else 0 for i in range(resolution + 1)]
            else:
                rv = sp.stats.norm(mean, stdDev)
                result = rv.cdf(np.linspace(0, 1, resolution + 1))
            cache[(mean, stdDev)] = result

        cdfs = cache[(mean, stdDev)]
        for row in range(resolution):
            data[row][col] = cdfs[row + 1] - cdfs[row]

    signals.progress(data, settings)


