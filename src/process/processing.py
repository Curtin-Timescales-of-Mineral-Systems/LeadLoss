from enum import Enum

from scipy.stats import stats

from model.column import Column
from src.model.settings.calculation import DiscordanceClassificationMethod
from utils import errorUtils
from process import calculations
import time

TIME_PER_TASK = 0.0

class ProgressType(Enum):
    CONCORDANCE = 0,
    ERRORS = 1,
    SAMPLING = 2


def process(signals, rows, importSettings, calculationSettings):
    completed = _calculateErrors(signals, rows)
    if not completed:
        return

    completed = _calculateConcordantAges(signals, rows, importSettings, calculationSettings)
    if not completed:
        return

    completed, bestAge = _performRimAgeSampling(signals, rows, calculationSettings)
    if not completed:
        return

    signals.completed(bestAge)


def _calculateErrors(signals, rows):
    timePerRow = TIME_PER_TASK/len(rows)
    for i, row in enumerate(rows):
        time.sleep(timePerRow)
        if signals.halt():
            signals.cancelled()
            return False

        uPb = row.importedCellsByCol[Column.U_PB_VALUE].value
        uPbError = row.importedCellsByCol[Column.U_PB_ERROR].value
        pbPb = row.importedCellsByCol[Column.PB_PB_VALUE].value
        pbPbError = row.importedCellsByCol[Column.PB_PB_ERROR].value

        # This is the super expensive operation
        row.uPb = errorUtils.ufloat(uPb, uPbError)
        row.pbPb = errorUtils.ufloat(pbPb, pbPbError)

        progress = (i + 1) / len(rows)
        signals.progress(ProgressType.ERRORS, progress, i)

    return True


def _calculateConcordantAges(signals, rows, importSettings, calculationSettings):
    timePerRow = TIME_PER_TASK/len(rows)
    for i, row in enumerate(rows):
        time.sleep(timePerRow)
        if signals.halt():
            signals.cancelled()
            return False

        if calculationSettings.discordanceClassificationMethod == DiscordanceClassificationMethod.PERCENTAGE:
            discordance = calculations.discordance(row.uPbValue(), row.pbPbValue())
            row.concordant = discordance < calculationSettings.discordancePercentageCutoff
        else:
            discordance = None
            row.concordant = calculations.isConcordantErrorEllipse(
                row.uPbValue(),
                row.uPbStDev(importSettings),
                row.pbPbValue(),
                row.pbPbStDev(importSettings),
                calculationSettings.discordanceEllipseSigmas
            )

        row.concordantAge = calculations.concordant_age(row.uPbValue(), row.pbPbValue()) if row.concordant else None
        signals.progress(ProgressType.CONCORDANCE, (i+1)/len(rows), i, row.concordantAge, discordance)

    return True


def _performRimAgeSampling(signals, rows, calculationSettings):
    # Actually compute the age distributions and statistics
    minAge = calculationSettings.minimumRimAge  # 500 * (10 ** 6)
    maxAge = calculationSettings.maximumRimAge  # 5000 * (10 ** 6)
    samples = calculationSettings.rimAgesSampled
    timePerSample = TIME_PER_TASK/samples

    concordantAges = [row.concordantAge for row in rows if row.concordant]

    results = []
    for i in range(samples):
        time.sleep(timePerSample)
        rimAge = minAge + i * ((maxAge - minAge) / (samples - 1))
        rimUPb = calculations.u238pb206_from_age(rimAge)
        rimPbPb = calculations.pb207pb206_from_age(rimAge)

        allReconstructedAges = []
        reconstructedAges = []
        for j, row in enumerate(rows):
            if signals.halt():
                signals.cancelled()
                return False, None

            if row.concordant:
                allReconstructedAges.append(None)
            else:
                reconstructedAge = calculations.discordant_age(rimUPb, rimPbPb, row.uPb, row.pbPb, 1)
                allReconstructedAges.append(reconstructedAge)

                if reconstructedAge is None:
                    reconstructedAges.append(0.0)
                else:
                    reconstructedAges.append(reconstructedAge.values[0])

        dValue, pValue = _calculateStatistics(concordantAges, reconstructedAges)
        results.append((rimAge, dValue, pValue))

        progress = (i + 1) / samples
        signals.progress(ProgressType.SAMPLING, progress, i, rimAge, allReconstructedAges, dValue, pValue)

    bestAge = _findOptimalAge(results)
    return True, bestAge


def _calculateStatistics(concordantAges, reconstructedAges):
    if not reconstructedAges or not concordantAges:
        return 0
    dValue, pValue = stats.ks_2samp(concordantAges, reconstructedAges)
    return dValue, pValue

def _findOptimalAge(results):
    minIndex, minArgs = min(enumerate(results), key=lambda v:v[1][1])

    startMinIndex = minIndex
    while startMinIndex > 0 and results[startMinIndex-1][1] == minArgs[1]:
        startMinIndex -=1

    endMinIndex = minIndex
    while endMinIndex < len(results) - 1 and results[endMinIndex+1][1] == minArgs[1]:
        endMinIndex += 1

    middleMinIndex = (endMinIndex + startMinIndex) // 2
    print(startMinIndex, middleMinIndex, endMinIndex)
    return results[middleMinIndex][0]
