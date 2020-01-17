#!/usr/bin/env python
import os
import sys
import glob
ROOT_DIR = os.getenv('PLASTICC_DIR')
sys.path.append(os.path.join(ROOT_DIR, 'plasticc'))
sys.path.append(os.path.join(ROOT_DIR, 'plasticc', 'plasticc'))
import h5py






def combine_hdf_files(save_dir, data_release, combined_savename):
    fnames = glob.glob(os.path.join(save_dir, 'features*hdf5'))
    fname_out = os.path.join(ROOT_DIR, 'plasticc', combined_savename)
    output_file = h5py.File(fname_out, 'w')
    # keep track of the total number of rows
    total_rows = 0
    for n, f in enumerate(fnames):
        print(f)
        f_hdf = h5py.File(os.path.join(save_dir, f), 'r')
        data = f_hdf[data_release]
        total_rows = total_rows + data.shape[0]
        if n == 0:
            # first file; fill the first section of the dataset; create with no max shape
            create_dataset = output_file.create_dataset(data_release, data=data, chunks=True, maxshape=(None,), compression='gzip')
            where_to_start_appending = total_rows
        else:
            # resize the dataset to accomodate the new data
            create_dataset.resize(total_rows, axis=0)
            create_dataset[where_to_start_appending:total_rows] = data
            where_to_start_appending = total_rows
        f_hdf.close()
    output_file.close()



def main():
    data_release = '20180901'
    field = 'DDF'
    model = '%'

    save_dir = os.path.join(ROOT_DIR, 'plasticc', 'Tables', 'features', 'hdf_features_{}_{}'.format(field, data_release))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    combine_hdf_files(save_dir, data_release, 'features_{}_{}.hdf5'.format(field, data_release))


if __name__ == '__main__':
    main()





