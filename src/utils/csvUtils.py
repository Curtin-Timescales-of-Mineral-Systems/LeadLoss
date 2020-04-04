import csv
from enum import Enum

from PyQt5.QtCore import QRegExp

from utils.config import *


#######################
## Column references ##
#######################

class ColumnReferenceType(Enum):
    LETTERS = "Letters"
    NUMBERS = "Numbers"


COLUMN_REFERENCE_TYPE_REGEXES = {
    ColumnReferenceType.LETTERS: QRegExp("[A-Z]+"),
    ColumnReferenceType.NUMBERS: QRegExp("[1-9]([0-9]*)")
}


def columnNumberToLetters(number, zeroIndexed):
    if isinstance(number, str):
        return number

    if zeroIndexed:
        number += 1

    letters = ""
    while True:
        letters = _numberToLetter(number % 26) + letters
        number = number // 26
        if number == 0:
            return letters


def columnLettersToNumber(letters, zeroIndexed):
    if isinstance(letters, int):
        return letters

    if isinstance(letters, str):
        letters = letters.replace(" ", "")
        number = 0
        for i, char in enumerate(letters):
            digit = _letterToNumber(char)
            exponent = len(letters) - i - 1
            number += digit * (26 ** exponent)
    if zeroIndexed:
        number -= 1
    return number

def convertColumnRef(ref, columnRefType, zeroIndexed):
    if columnRefType is ColumnReferenceType.NUMBERS:
        return columnLettersToNumber(ref, zeroIndexed)
    if columnRefType is ColumnReferenceType.LETTERS:
        return columnNumberToLetters(ref, zeroIndexed)
    raise Exception("Unexpected ColumnReferenceType: " + str(columnRefType))

def _letterToNumber(letter):
    return ord(letter) - 64


def _numberToLetter(number):
    return chr(number + 64)


#########################
## Reading and writing ##
#########################

def read_input(input_file, settings):
    with open(input_file, newline='', encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file, delimiter=settings.delimiter, quotechar='|')
        lines = [line for line in reader]

        if settings.hasHeaders:
            rows = lines[1:]
            headers = lines[0]
        else:
            rows = lines
            headers = None

    largestColumnNumberAskedFor = max(settings.getDisplayColumns())

    for line in lines:
        if largestColumnNumberAskedFor >= len(line):
            largestColumnRefAvailable = convertColumnRef(len(line), settings.columnReferenceType, False)
            largestColumnRefAskedFor = convertColumnRef(largestColumnNumberAskedFor+1, settings.columnReferenceType, False)
            raise Exception(
                "Invalid column reference. Asked for column " + str(largestColumnRefAskedFor) + " but the CSV file only"
                " goes up to column " + str(largestColumnRefAvailable) + "."
            )

    return headers, rows


def write_output(headers, rows, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=CSV_DELIMITER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


###########
## Other ##
###########

"""
def perform_all_calculations(headers, rows, figure_dir, progress_callback):
    error_strings = ["-" + stringUtils.ERROR_STR_OUTPUT, "+" + stringUtils.ERROR_STR_OUTPUT]
    headers.extend(
        ["Recon. age"] + error_strings + ["Recon. U238/Pb206"] + error_strings + ["Recon. Pb207/Pb206"] + error_strings)

    progress_callback(0)
    for i, row in enumerate(rows[1:]):
        progress_callback(i + 1)
        try:
            _perform_calculations(i + 1, row, figure_dir)
        except ValueError as e:
            stringUtils.print_warning("\rIgnoring row " + str(i) + ": " + str(e))
    progress_callback(len(rows))

def _parse(column_ref, row, row_number, column_name=None):
    column_number = stringUtils.get_column_number(column_ref)
    string = row[column_number]
    try:
        return float(string)
    except:
        column_text = column_name if column_name else ("column " + column_number)
        raise ValueError(
            "Invalid value '" + string + "' for '" + column_text + "' in row " + str(row_number) + " column " + str(
                column_number))
"""
