# Utils


def constrain(val, upper_bound, lower_bound):
    """ Constrains the value between upper_bound and lower_bound.

    Args:
        val
        upper_bound
        lower_bound

    Returns:
        val
    """
    if val > upper_bound:
        return upper_bound
    elif val < lower_bound:
        return lower_bound
    else:
        return val
