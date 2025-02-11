![Curtin University: Timescales of Minerals Systems](resources/logo-linear.png)

## Lead-loss

Lead-Loss is a Python-based tool designed to estimate the most likely time of lead loss in discordant zircon samples. The tool provides a user-friendly GUI interface.

## Download and Installation

**Option 1: Standalone Executable**
We provide standalone executables for Windows and MacOS here:
https://github.com/MatthewDaggitt/LeadLoss/releases
Simply download and run the appropriate file for your system-no installation required.

**Option 2: Python Environment**
For Python users, you may clone the repository and install dependencies manually using:
git clone https://github.com/Curtin-Timescales-of-Mineral-Systems/LeadLoss.git
cd LeadLoss
pip install -3 requirements.txt

## Usage

**Option 1: Standalone Executable**
1. Download the executable from [the releases page](https://github.com/MatthewDaggitt/LeadLoss/releases)
2. Follow the Input Requirements below to prepare your CSV file.
3. Launch the executable and follow the GUI prompts.

**Option 2: Python Environment**
1. Clone the repository and set up the environment as described above.
2. Run the GUI using the following command:
python application.py

## Input Requirements

The program requires a Comma Separated Value (.csv) file with each row representing a single spot analysis. The CSV file must include the following columns:

238U/206Pb ratio
238U/206Pb uncertainty
207Pb/206Pb ratio
207Pb/206Pb uncertainty

During data import, the user will be prompted to specify the column names or indices corresponding to the required values (e.g., A, B, C, D or 1, 2, 3, 4). Optional columns, such as sample names for batch processing, may also be included. Uncertainties may be specified as absolute values or percentages at the 1σ or 2σ confidence levels.

## Outputs

The program generates the following outputs:
Optimal Pb-loss age estimates with 95% confidence intervals
p-values and D-values from the KS test
Individual Monte Carlo sampling results

## Troubleshooting

If the standalone GUI does not work, ensure you have downloaded the correct version for your operating system. For Python users, make sure all dependencies are correctly installed using the requirements.txt file. If you continue to experience issues, please open an issue on GitHub.

### Citation

If you use LeadLoss in your research, please cite:
Mathieson, L. (2024). LeadLoss: A Python Tool for Modelling the Timing of Lead Loss in Zircon (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.14039113
