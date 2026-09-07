from enum import Enum


class ConcordiaRatioSpace(Enum):
    TERA_WASSERBURG = "Tera-Wasserburg"
    WETHERILL = "Wetherill"

    def __eq__(self, other):
        return self.value == getattr(other, "value", other)


class ConcordiaSpaceSelection(Enum):
    SAME_AS_INPUT = "Same as input data"
    TERA_WASSERBURG = "Tera-Wasserburg"
    WETHERILL = "Wetherill"

    def __eq__(self, other):
        return self.value == getattr(other, "value", other)


def ratio_space_from_value(value, default=ConcordiaRatioSpace.TERA_WASSERBURG):
    value = getattr(value, "value", value)
    if value is None:
        return default
    text = str(value).strip().lower()
    if text == "wetherill":
        return ConcordiaRatioSpace.WETHERILL
    if text in {"tera-wasserburg", "tera wasserburg", "tw"}:
        return ConcordiaRatioSpace.TERA_WASSERBURG
    return default


def space_selection_from_value(value, default=ConcordiaSpaceSelection.SAME_AS_INPUT):
    value = getattr(value, "value", value)
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"same as input data", "same as input", "input"}:
        return ConcordiaSpaceSelection.SAME_AS_INPUT
    if text == "wetherill":
        return ConcordiaSpaceSelection.WETHERILL
    if text in {"tera-wasserburg", "tera wasserburg", "tw"}:
        return ConcordiaSpaceSelection.TERA_WASSERBURG
    return default


def resolve_space_selection(selection, input_space):
    selection = space_selection_from_value(selection)
    input_space = ratio_space_from_value(input_space)
    if selection == ConcordiaSpaceSelection.SAME_AS_INPUT:
        return input_space
    if selection == ConcordiaSpaceSelection.WETHERILL:
        return ConcordiaRatioSpace.WETHERILL
    return ConcordiaRatioSpace.TERA_WASSERBURG
