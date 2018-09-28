# Updated to match RK's mapping on 20180715 - GN

def get_sntypes():
    sntypes_map = {1: 'SNIa-Normal',
                   2: 'SNCC-II', 
                   12: 'SNCC-II', 
                   14: 'SNCC-II', 
                   3: 'SNCC-Ibc', 
                   13: 'SNCC-Ibc', 
                   5: 'SNCC-Ibc',
                   6: 'SNCC-II', 
                   41: 'SNIa-91bg',
                   43: 'SNIa-x',
                   51: 'Kilonova',
                   60: 'SLSN-I', 
                   61: 'PISN', 
                   62: 'ILOT', 
                   63: 'CART', 
                   64: 'TDE',
                   70: 'AGN',
                   80: 'RRLyrae', 
                   81: 'Mdwarf', 
                   83: 'EBE',
                   84: 'Mira', 
                   90: 'uLens-Binary', 
                   91: 'uLens-Point',
                   92: 'uLens-String',
                   93: 'uLens-Point',
                   94: 'uLens-Point',
                   99: 'Rare'}
    return sntypes_map


def aggregate_sntypes(reverse=False):
    if reverse:
        aggregate_map = {99: (61, 62, 63, 90, 92),
                         1: (1,),
                         5: (3,13),
                         6: (2, 12, 14),
                         41: (41,),
                         43: (43,),
                         51: (51,),
                         60: (60,),
                         64: (64,),
                         70: (70,),
                         80: (80,),
                         81: (81,),
                         83: (83,),
                         84: (84,),
                         94: (91, 93)
                         }
    else:
        aggregate_map = {1: 1,
                         2: 6, 
                         3: 5,
                         12: 6,
                         13: 5,
                         14: 6,
                         41: 41, 
                         43: 43,
                         51: 51,
                         60: 60,
                         61: 99,
                         62: 99,
                         63: 99,
                         64: 64,
                         70: 70,
                         80: 80,
                         81: 81,
                         83: 83,
                         84: 84,
                         90: 99,
                         91: 94,
                         92: 99,
                         93: 94,
                         }

    return aggregate_map


def remove_field_name(a, name):
    names = list(a.dtype.names)
    if name in names:
        names.remove(name)
    b = a[names]
    return b
