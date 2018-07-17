import os
import numpy as np
import matplotlib.pyplot as plt

models = ['SNIa', 'SNII', 'SNIbc', 'SNII-pca', 'SNIbc', 'SNIIn', 'SNIa-91bg', 'SNIax', 'Point-Ia', 'GW170817', 'Kilonova_2', 'Magnetar', 'PISN', 'ILOT', 'CART', 'TDE', 'RRLyrae', 'Mdwarf', 'AGN', 'Eclip. Bin.', 'BSR', 'String']
nobjects_ddf = [14013, 4516, 2523, 5533, 1224, 531, 423 , 866 , 3, 0, 1, 149, 8, 21, 60, 98, 10, 59, 150, 150, 0, 0]
nobjects_wfd = [1498486, 394613, 64629, 198901, 96020, 46288, 36418, 56901, 210, 20, 39, 32313, 1044, 1456, 8559, 12656, 48346, 14372, 74205, 6046, 1, 0]


def autolabel(rects, ax, sigfig=4):
    """ Labels on top of bar charts. """
    # Get y-axis height to calculate label position from.
    (y_bottom, y_top) = ax.get_ylim()
    y_height = y_top - y_bottom

    for rect in rects:
        height = rect.get_height()

        # Fraction of axis height taken up by this rectangle
        p_height = (height / y_height)

        # If we can fit the label above the column, do that; otherwise, put it inside the column.
        if p_height > 0.95:  # arbitrary; 95% looked good to me.
            label_position = height - (y_height * 0.1)
        else:
            label_position = height + (y_height * 0.01)

        ax.text(rect.get_x() + rect.get_width()/2., label_position, '%s' % int(height), ha='center', va='bottom', rotation=90, fontsize=14)


stat_label = 'Number of objects'
stat_name = 'nobjects_wfd'
stat = nobjects_wfd

# fig, ax = plt.subplots(figsize=(20,7))
# y, yerr = stat, np.zeros(len(stat))
# rects = ax.bar(range(len(models)), y, yerr=yerr, align='center')
# ax.set_xticks(range(len(models)))
# ax.set_xticklabels(models)
# plt.xticks(rotation=90, fontsize=14)
# # ax.set_xlabel('model_name')
# ax.set_ylabel(stat_label, fontsize=14)
# plt.yticks(fontsize=14)
# ax.set_ylim(bottom=0)
# autolabel(rects, ax)
# plt.tight_layout()
# fig.savefig(stat_name, fontsize=18)
# plt.show()

stat_label1 = 'Number of objects, DDF'
stat_label2 = 'Number of objects, WFD'
stat_name = 'nobjects'
stat1 = nobjects_ddf
stat2 = nobjects_wfd

ind = np.arange(len(models))
width = 0.4

# Make two bars with two axes
fig, ax1 = plt.subplots(figsize=(20,10))
y, yerr = stat1, np.zeros(len(stat1))
rects1 = ax1.bar(ind, y, -width, yerr=yerr, align='edge', color='#1f77b4')
plt.yticks(fontsize=19)

ax2 = ax1.twinx()
y, yerr = stat2, np.zeros(len(stat2))
rects2 = ax2.bar(ind+width, y, -width, yerr=yerr, align='edge', color='#ff7f0e')

ax1.set_ylabel(stat_label1, fontsize=22)
ax2.set_ylabel(stat_label2, fontsize=22, rotation=270, labelpad=25)
plt.yticks(fontsize=19)

ax1.set_ylim(bottom=0)
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models)
ax1.legend((rects1[0], rects2[0]), ('DDF', 'WFD'), fontsize=26)

autolabel(rects1, ax1)
autolabel(rects2, ax2)

plt.sca(ax1)
plt.xticks(rotation=90, fontsize=19)

plt.tight_layout()
fig.savefig(stat_name + '.pdf')
plt.show()


