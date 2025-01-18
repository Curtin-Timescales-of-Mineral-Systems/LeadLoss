import math
import time
from enum import Enum

import numpy as np
import scipy as sp
from scipy.stats import stats

from model.monteCarloRun import MonteCarloRun
from model.monteCarloRunWetherill import MonteCarloRunWetherill
from process import calculations
from process import calculationsWetherill as calcW
from model.settings.calculation import DiscordanceClassificationMethod
from utils import config

TIME_PER_TASK = 0.0


class ProgressType(Enum):
    CONCORDANCE = 0,
    SAMPLING = 1,
    OPTIMAL = 2,

def processSamples(signals, samples):
    for sample in samples:
        completed, skip_reason = _processSample(signals, sample)
        if not completed:
            if skip_reason:
                signals.skipped(sample.name, skip_reason)  # Send skip reason back to main thread
            continue
    signals.completed()

def _processSample(signals, sample):
    """
    Master function that chooses either the Tera–W pipeline or the Wetherill pipeline
    based on sample.calculationSettings.concordiaMode.
    """
    settings = sample.calculationSettings
    mode = getattr(settings, "concordiaMode", "TW")

    if mode == 'Wetherill':
        return _processSampleWetherill(signals, sample)

    else:
        completed, skip_reason = _calculateConcordantAges(signals, sample)
        if not completed:
            return False, skip_reason

        completed, skip_reason = _performRimAgeSampling(signals, sample)
        if not completed:
            return False, skip_reason

        return True, None

def _processSampleWetherill(signals, sample):
    completed, skip_reason = _calculateConcordantAgesWetherill(signals, sample)
    if not completed:
        return False, skip_reason
    completed, skip_reason = _performRimAgeSamplingWetherill(signals, sample)
    if not completed:
        return False, skip_reason

    return True, None

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
            return False, "processing halted by user"

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
    return True, None  # Indicate success

def _calculateConcordantAgesWetherill(signals, sample):
    sampleNameText = f" for '{sample.name}'" if sample.name else ""
    signals.newTask("Classifying points (Wetherill)" + sampleNameText + "...")

    settings = sample.calculationSettings
    timePerRow = TIME_PER_TASK / len(sample.validSpots)

    concordancy = []
    discordances = []

    for i, spot in enumerate(sample.validSpots):
        signals.progress(ProgressType.CONCORDANCE, i / len(sample.validSpots))
        time.sleep(timePerRow)
        if signals.halt():
            signals.cancelled()
            return False, "processing halted by user"

        ratio206_238 = spot.pb206U238Value
        error206_238 = spot.pb206U238Error
        ratio207_235 = spot.pb207U235Value
        error207_235 = spot.pb207U235Error

        if settings.discordanceClassificationMethod == DiscordanceClassificationMethod.PERCENTAGE:
            discordance = calcW.discordance_wetherill(ratio207_235, ratio206_238)
            concordant = (discordance < settings.discordancePercentageCutoff)
            discordances.append(discordance)

        else:
            discordance = None
            concordant = calcW.isConcordantErrorEllipseWetherill(
                ratio207_235, error207_235,
                ratio206_238, error206_238,
                settings.discordanceEllipseSigmas
            )
        concordancy.append(concordant)
        discordances.append(discordance)

    sample.updateConcordance(concordancy, discordances)

    signals.progress(ProgressType.CONCORDANCE, 1.0, sample.name, concordancy, discordances)
    return True, None  # Indicate success

def _performRimAgeSampling(signals, sample):
    sampleNameText = " for '" + sample.name + "'" if sample.name else ""
    signals.newTask("Sampling Pb-loss age distributions" + sampleNameText + "...")

    settings = sample.calculationSettings

    # Get the concordant samples
    concordantSpots = [spot for spot in sample.validSpots if spot.concordant]
    discordantSpots = [spot for spot in sample.validSpots if not spot.concordant]

    # If there are no concordant spots
    if not concordantSpots:
        return False, "no concordant spots"
    # If there are no discordant spots
    if not discordantSpots:
        return False, "no discordant spots"
    # If there are fewer than 3 discordant spots
    if len(discordantSpots) <= 2:
        return False, "fewer than 3 discordant spots"

    # Proceed with processing
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

    # Continue processing
    for j in range(stabilitySamples):
        if signals.halt():
            signals.cancelled()
            return False, "processing halted by user"

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
    return True, None  # Indicate success

def _performRimAgeSamplingWetherill(signals, sample):
    sampleNameText = f" for '{sample.name}'" if sample.name else ""
    signals.newTask("Sampling Pb-loss age distributions (Wetherill)" + sampleNameText + "...")
    
    settings = sample.calculationSettings

    concordantSpots = [s for s in sample.validSpots if s.concordant]
    discordantSpots = [s for s in sample.validSpots if not s.concordant]

    if not concordantSpots:
        return False, "no concordant spots"
    if not discordantSpots:
        return False, "no discordant spots"
    if len(discordantSpots) <= 2:
        return False, "fewer than 3 discordant spots"

    # 2) Monte Carlo runs
    stabilitySamples = settings.monteCarloRuns
    random = np.random.RandomState()

    # (a) Concordant Wetherill ratio draws
    conc_206_238_draws = np.transpose([
        random.normal(spot.pb206U238Value, spot.pb206U238Error, stabilitySamples)
        for spot in concordantSpots
    ])
    conc_207_235_draws = np.transpose([
        random.normal(spot.pb207U235Value, spot.pb207U235Error, stabilitySamples)
        for spot in concordantSpots
    ])

    # (b) Discordant ratio draws
    disc_206_238_draws = np.transpose([
        random.normal(spot.pb206U238Value, spot.pb206U238Error, stabilitySamples)
        for spot in discordantSpots
    ])
    disc_207_235_draws = np.transpose([
        random.normal(spot.pb207U235Value, spot.pb207U235Error, stabilitySamples)
        for spot in discordantSpots
    ])

    # 3) For each Monte Carlo iteration, store the “run” object
    for j in range(stabilitySamples):
        if signals.halt():
            signals.cancelled()
            return False, "processing halted by user"

        runConcordant206_238 = conc_206_238_draws[j]
        runConcordant207_235 = conc_207_235_draws[j]
        runDiscordant206_238 = disc_206_238_draws[j]
        runDiscordant207_235 = disc_207_235_draws[j]

        run = MonteCarloRunWetherill(
            run_number = j,
            sample_name = sample.name,
            # pass the draws for the 'concordant' spots
            concordant_206_238 = conc_206_238_draws[j],
            concordant_207_235 = conc_207_235_draws[j],
            # pass the draws for the 'discordant' spots
            discordant_206_238 = disc_206_238_draws[j],
            discordant_207_235 = disc_207_235_draws[j],
        )

        _performSingleRunWetherill(settings, run)
        sample.addMonteCarloRun(run)

        progress = (j + 1) / stabilitySamples
        signals.progress(ProgressType.SAMPLING, progress, sample.name, run)

        if j % 5 == 0 or j == stabilitySamples - 1:
            _calculateOptimalAge(signals, sample, progress)
    return True, None

def _performSingleRun(settings, run):
    # Generate the lead loss age samples
    for age in settings.rimAges():
        run.samplePbLossAge(age, settings.dissimilarityTest, settings.penaliseInvalidAges)
    run.calculateOptimalAge()
    run.createHeatmapData(settings.minimumRimAge, settings.maximumRimAge, config.HEATMAP_RESOLUTION)

def _performSingleRunWetherill(settings, run):
    # print("DEBUG => in _performSingleRunWetherill, rim ages =", settings.rimAges())
    for age in settings.rimAges():
        # print(f"DEBUG => calling samplePbLossAgeWetherill at age={age}")
        run.samplePbLossAgeWetherill(age, settings.dissimilarityTest, settings.penaliseInvalidAges)
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