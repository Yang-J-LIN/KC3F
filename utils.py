# Utils


def constrain(val, upper_bound, lower_bound):
    if val > upper_bound:
        return upper_bound
    elif val < lower_bound:
        return lower_bound
    else:
        return val
