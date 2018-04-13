

def get_sntypes():
    sntypes_map = {1: 'SN1a', 2: 'CC', 3: 'SNIbc', 4: 'IIpca', 5: 'New_CC', 41: 'Ia-91bg', 42: 'Ia-91bg-Jones',
                   45: 'pointIa', 50: 'Kilonova', 60: 'Magnetar', 61: 'PISN', 62: 'ILOT', 63: 'CART', 64: 'TDE',
                   80: 'RRLyrae', 81: 'Mdwarf', 82: 'Mira', 90: 'BSR', 91: 'String',
                   102: 'CC', 141: 'Ia-91bg', 200: 'Other'}
    return sntypes_map


def aggregate_sntypes(reverse=False):
    # aggregate_map = {'Ia': ('SN1a', 'Ia-91bg-Santiago', 'Ia-91bg-Jones'),
    #                  'CC': ('CC', 'New_CC', 'IIn'),
    #                  'Other': ('Kilonova', 'ILOT', 'BSR', 'String'),
    #                  'SNIbc': ('SNIbc',),
    #                  'pointIa': ('pointIa', ),
    #                  'Magnetar': ('Magnetar',),
    #                  'PISN': ('PISN',),
    #                  'CART': ('CART',),
    #                  '64': ('64',),
    #                  'RRLyrae': ('RRLyrae',),
    #                  'Mdwarf': ('Mdwarf',),
    #                  'Mira': ('Mira',),
    #                  }

    if reverse:
        aggregate_map = {2: (2, 3, 4, 5),
                         41: (41, 42),
                         1: (1,),
                         45: (45,),
                         50: (50,),
                         60: (60,),
                         61: (61,),
                         62: (62,),
                         63: (63,),
                         64: (64,),
                         80: (80,),
                         81: (81,),
                         82: (82,),
                         90: (90,),
                         91: (91,),
                         }
    else:
        aggregate_map = {2: 2, 3: 2, 4: 2, 5: 2,
                         41: 41, 42: 41,
                         1: 1,
                         45: 45,
                         50: 50,
                         60: 60,
                         61: 61,
                         62: 62,
                         63: 63,
                         64: 64,
                         80: 80,
                         81: 81,
                         82: 82,
                         90: 90,
                         91: 91,
                         }
        # Other is BSR (Binary Star remnant), ILOT, PISN

    return aggregate_map
