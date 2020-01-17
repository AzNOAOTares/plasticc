# Updated to match RK's mapping on 20180715 - GN
#
# def get_sntypes():
#     sntypes_map = {1: 'SN1a', 2: 'CC-II', 3: 'CC-Ibc', 6: 'SNII', 12: 'IIpca', 13: 'SNIbc', 14: 'IIn',
#                    41: 'Ia-91bg', 42: 'Ia-91bg-Jones', 43: 'Iax', 45: 'pointIa',
#                    50: 'Kilonova-GW170817', 51: 'Kilonova-Kasen', 60: 'Magnetar', 61: 'PISN', 62: 'ILOT', 63: 'CART', 64: 'TDE',
#                    80: 'RRLyrae', 81: 'Mdwarf', 82: 'AGN', 83: 'PHOEBE', 84: 'Mira', 90: 'BSR', 91: 'String'}
#     return sntypes_map
#
#
# def aggregate_sntypes(reverse=False):
#     if reverse:
#         aggregate_map = {6: (2, 12, 14),
#                          13: (13, 3),
#                          41: (41,),
#                          1: (1,),
#                          43: (43,),
#                          45: (45,),
#                          50: (50,),
#                          51: (51,),
#                          60: (60,),
#                          61: (61,),
#                          62: (62,),
#                          63: (63,),
#                          64: (64,),
#                          80: (80,),
#                          81: (81,),
#                          82: (82,),
#                          83: (83,),
#                          84: (84,),
#                          90: (90,),
#                          91: (91,),
#                          }
#     else:
#         aggregate_map = {2: 6, 12: 6, 14: 6,
#                          3: 13, 13: 13,
#                          41: 41, 42: 'ignore',
#                          1: 1,
#                          43: 43,
#                          45: 45,
#                          50: 50,
#                          51: 51,
#                          60: 60,
#                          61: 61,
#                          62: 62,
#                          63: 63,
#                          64: 64,
#                          80: 80,
#                          81: 81,
#                          82: 82,
#                          83: 83,
#                          84: 84,
#                          90: 90,
#                          91: 91,
#                          }
#
#     return aggregate_map
#
#
# def remove_field_name(a, name):
#     names = list(a.dtype.names)
#     if name in names:
#         names.remove(name)
#     b = a[names]
#     return b


def get_sntypes():
    sntypes_map = {11: 'SNIa-Normal',
                   2: 'SNCC-II',
                   12: 'SNCC-II',
                   14: 'SNCC-II',
                   3: 'SNCC-Ibc',
                   13: 'SNCC-Ibc',
                   41: 'Ia-91bg',
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
                   90: 'uLens-binary',
                   91: 'uLens-point',
                   92: 'uLens-string',
                   93: 'uLens-point',
                   99: 'Rare'}
    return sntypes_map


def aggregate_sntypes(reverse=False):
    if reverse:
        aggregate_map = {99: (61, 62, 63, 90, 92),
                         11: (11,),
                         3: (3,13),
                         2: (2, 12, 14),
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
                         91: (91, 93)}
    else:
        aggregate_map = {11: 11,
                         2: 2,
                         3: 3,
                         12: 2,
                         13: 3,
                         14: 2,
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
                         91: 91,
                         92: 99,
                         93: 91,
                         }

    return aggregate_map


def remove_field_name(a, name):
    names = list(a.dtype.names)
    if name in names:
        names.remove(name)
    b = a[names]
    return b
