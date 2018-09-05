import numpy as np
import matplotlib.pyplot as plt
from plasticc import get_data
from plasticc import helpers


data_release = 'ZTF_20180716'
field = 'MSIP'
snid = '%'
model = 1
base = '%'
pb = 'r'

sntypes_map = helpers.get_sntypes()

getter = get_data.GetData(data_release)

for model in [1, 2, 12, 14, 13, 41, 43, 45, 50, 51, 60, 61, 62, 63, 64, 70]:
    print(sntypes_map[model], model)
    result = getter.get_lcs_data(columns=['objid', 'ptrobs_min', 'ptrobs_max', 'peakmjd', 'hostgal_photoz', 'mwebv', 'sim_dlmu'], field=field, snid=snid, model=model, base=base, limit=10, offset=0)
    head, phot = next(result)
    objid, ptrobs_min, ptrobs_max, peakmjd, z, mwebv, dlmu = head
    lc = getter.convert_pandas_lc_to_recarray_lc(phot)
    d = 10 ** (dlmu/5 + 1)
    flux = phot[pb]['FLUXCAL'] * 4 * np.pi * d**2
    flux = flux.clip(min=0)
    mag = -2.5 * np.log10(flux) + 27.5
    time = (phot[pb]['MJD'] - peakmjd)/(1 + z)
    flux = flux[(time < 100) & (time > -40)]
    mag = mag[(time < 100) & (time > -40)]
    time = time[(time < 100) & (time > -40)]
    plt.plot(time, mag, marker='.', label=sntypes_map[model])

plt.legend()
plt.xlabel('mjd')
plt.ylabel('$log_{10}(flux)$')
ax = plt.gca()
ax.invert_yaxis()
plt.savefig('example_light_curves.pdf')
plt.show()

