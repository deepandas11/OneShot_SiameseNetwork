


def gt_label_gen(a,b):
    """
    :params a,b: Two integers/Class categories.
    :return 0/1: If they belong to same class, return 1, else 0
    """
    return int(not bool(a-b))