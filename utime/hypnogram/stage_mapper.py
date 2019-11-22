import re
import numpy as np

from utime import defaults


def raise_match_error(ss, reason):
    raise ValueError("Could not consistently decode variable sleep stage "
                     "string '{:s}' - {:s}".format(ss, reason))


def check_number_match(ss):
    possible_match, match_value = False, None

    # Check for number referring to the sleep stage
    numbers = list(map(int, re.findall(r"\d+", ss)))
    if len(numbers) not in (0, 1):
        raise_match_error(ss, "Found multiple numbers in string")
    elif len(numbers) == 1:
        num = numbers[0]
        valid_map = {1: defaults.NON_REM_STAGE_1[0],
                     2: defaults.NON_REM_STAGE_2[0],
                     3: defaults.NON_REM_STAGE_3[0],
                     4: defaults.NON_REM_STAGE_3[0]}
        assert np.all(np.in1d(list(valid_map.values()),
                              list(defaults.stage_string_to_class_int.keys())))
        if num in valid_map:
            possible_match = True
            match_value = valid_map[num]
        else:
            raise_match_error(ss, "Found invalid number {} in string".format(num))

    return possible_match, match_value


def check_wake_match(ss):
    possible_match, match_value = False, None
    valid_substrings = ("WAKE", "WK", "W")
    in_string = [s in ss for s in valid_substrings]
    if any(in_string):
        possible_match, match_value = True, defaults.AWAKE[0]
    return possible_match, match_value


def check_REM_match(ss):
    possible_match, match_value = False, None
    valid_substrings = ("REM", "RAPID", "EYE", "R-E-M", "R.E.M.", "R")
    in_string = [s in ss for s in valid_substrings]
    if any(in_string):
        possible_match, match_value = True, defaults.REM[0]
    return possible_match, match_value


def check_unknown_match(ss):
    possible_match, match_value = False, None
    valid_substrings = ("UNKNOWN", "OTHER", "MOVEMENT", "?", "NA", "MOVE",
                        "MOVING", "MT")
    in_string = [s in ss for s in valid_substrings]
    if any(in_string):
        possible_match, match_value = True, defaults.UNKNOWN[0]
    return possible_match, match_value


def standardize_stage_string(stage_string):
    """
    Attempts to map a string representing a sleep stage of which some
    (unambiguous) variability is allowed to a fixed, standardised
    string representation

    Standardized strings:
        "W" (awake)
        "N1" (non-rem sleep stage 1)
        "N2" (non-rem sleep stage 2)
        "N3" (non-rem sleep stage 3)
        "REM" (REM sleep)
        "UNKNOWN" (all other)

    Args:
        stage_string: A string representing a sleep stage

    Returns:
        string, A standardized string representing a sleep stage
    """
    ss = stage_string.strip().upper()

    # Check various types of matches
    matches = []
    for match_func in (check_number_match, check_wake_match,
                       check_REM_match, check_unknown_match):
        possible_match, match_value = match_func(ss)
        if possible_match:
            matches.append(match_value)

    # If exactly 1 match was found, return this, otherwise raise an error
    n_matches = len(matches)
    if n_matches == 1:
        match = matches[0]
        print("[OBS]: Mapping variable stage string '{:s}' to stage '{}' "
              "(class int {})".format(stage_string, match,
                                      defaults.stage_string_to_class_int[match]))
        return match
    elif n_matches == 0:
        raise_match_error(ss, "Found no valid matches.")
    else:
        raise_match_error(ss, "Found multiple ({}) "
                              "valid matches {}.".format(n_matches, matches))


def stage_string_to_class(stage_string):
    return defaults.stage_string_to_class_int[stage_string.upper()]


def create_variable_ann_to_class_int_dict(annotations):
    import numpy as np
    unique_ann = np.unique(annotations)
    mapping = {s: stage_string_to_class(standardize_stage_string(s)) for s in unique_ann}
    return mapping
