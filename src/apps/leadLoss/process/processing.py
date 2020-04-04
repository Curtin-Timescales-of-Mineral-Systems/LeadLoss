from enum import Enum

from scipy.stats import stats

from apps.leadLoss.model.column import Column
from utils import errorUtils, calculations


class ProgressType(Enum):
    CONCORDANCE = 0,
    ERRORS = 1,
    SAMPLING = 2


def process(signals, rows, importSettings, calculationSettings):
    completed = _calculateErrors(signals, rows)
    if not completed:
        return

    completed = _calculateConcordantAges(signals, rows, calculationSettings)
    if not completed:
        return

    completed = _performRimAgeSampling(signals, rows, calculationSettings)
    if not completed:
        return

    signals.completed(None)


def _calculateErrors(signals, rows):
    for i, row in enumerate(rows):
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


def _calculateConcordantAges(signals, rows, calculationSettings):
    for i, row in enumerate(rows):
        if signals.halt():
            signals.cancelled()
            return False

        discordance = calculations.discordance(row.uPbValue(), row.pbPbValue())
        row.concordant = discordance < calculationSettings.discordancePercentageCutoff
        row.concordantAge = calculations.concordant_age(row.uPbValue(), row.pbPbValue()) if row.concordant else None
        signals.progress(ProgressType.CONCORDANCE, (i+1)/len(rows), i, row.concordantAge, discordance)

    return True


def _performRimAgeSampling(signals, rows, calculationSettings):
    # Actually compute the age distributions and statistics
    minAge = calculationSettings.minimumRimAge  # 500 * (10 ** 6)
    maxAge = calculationSettings.maximumRimAge  # 5000 * (10 ** 6)
    samples = calculationSettings.rimAgesSampled

    concordantAges = [row.concordantAge for row in rows if row.concordant]

    outputs = []
    for i in range(samples):
        rimAge = minAge + i * ((maxAge - minAge) / (samples - 1))
        rimUPb = calculations.u238pb206_from_age(rimAge)
        rimPbPb = calculations.pb207pb206_from_age(rimAge)

        discordantAges = []
        for j, row in enumerate(rows):
            if signals.halt():
                signals.cancelled()
                return False

            if row.concordant:
                reconstructedAge = None
            else:
                reconstructedAge = calculations.discordant_age(rimUPb, rimPbPb, row.uPb, row.pbPb, 1)
            discordantAges.append(reconstructedAge)

        statistic = _calculateStatistics(concordantAges, discordantAges)
        outputs.append((rimAge, discordantAges, statistic))

        progress = (i + 1) / samples
        signals.progress(ProgressType.SAMPLING, progress, i)
    return True


def _calculateStatistics(concordantAges, reconstructedAges):
    discordantAges = []
    for reconstructedAge in reconstructedAges:
        if reconstructedAge is not None:
            discordantAges.append(reconstructedAge.values[0])

    if not discordantAges or not concordantAges:
        return 0

    pValue = stats.ks_2samp(concordantAges, discordantAges)[1]
    print(pValue)
    return pValue
