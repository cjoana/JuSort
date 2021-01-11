"""
Based on raw recordings detect spikes, calculate features and do automatic
clustering with gaussian mixture models.
"""

import os

import spike_sort as sort
from spike_sort.io.filters import PyTablesFilter
import spike_sort.ui.manual_sort
import numpy as np
import active_vision.utils as avutils
from active_vision.fileio.lvdread import LVDReader
import h5py
from conf.conf_files import projectdir, prepdir, rawdir, datadir, spkdir
import matplotlib.pyplot as plt

rawdir2 = "/datasets/SUPERosaka/RAWDATA"
# rawdir =rawdir2


alldatasets = [
		["V1", "..."],

	]

dataset = alldatasets[0]
data, subject, sess, rec,blk, pc =dataset

fn_raw = avutils.find_filenames(rawdir, subject, sess, rec, 'lvd', pc=pc)[0]
fn_features = "{dir}/hdf5_files/{sbj}/features_{sess}_rec{rec}_blk{blk}_{data}_{sbj}.hdf5".format(dir=datadir, sbj=subject,
																						  data=data,
																						  blk=blk, rec=rec, sess=sess)

f = h5py.File(fn_features, 'r')
blk_start = f["blk_start"][...]
blk_end = f["blk_end"][...]

print "file found, loading ..."
lvd_reader = LVDReader(fn_raw)

chans =  14 # list(np.arange(0,24))
data = lvd_reader.get_data(channel=chans, timerange=(blk_start, blk_start + 10))[0]
# data = lvd_reader.get_data(channel=23 - channel  ,  timerange=(t_start, t_end))[0]
nd = len(data)
print "Done.", nd

# h5filter = PyTablesFilter(fn_raw, 'a')
# sp = h5filter.read_sp()
spt = sort.extract.detect_spikes(data, thresh='auto')



"""


if __name__ == "__main__":
	h5_fname = os.path.join(DATAPATH, "tutorial.h5")
	h5filter = PyTablesFilter(h5_fname, 'a')

	dataset = "/SubjectA/session01/el1"
	sp_win = [-0.2, 0.8]

	sp = h5filter.read_sp(dataset)
	spt = sort.extract.detect_spikes(sp, contact=3, thresh='auto')

	spt = sort.extract.align_spikes(sp, spt, sp_win, type="max", resample=10)
	sp_waves = sort.extract.extract_spikes(sp, spt, sp_win)
	features = sort.features.combine(
		(sort.features.fetP2P(sp_waves),
		 sort.features.fetPCA(sp_waves)),
		norm=True
	)

	clust_idx = sort.cluster.cluster("gmm", features, 4)

	spike_sort.ui.plotting.plot_features(features, clust_idx)
	spike_sort.ui.plotting.figure()
	spike_sort.ui.plotting.plot_spikes(sp_waves, clust_idx, n_spikes=200)

	spike_sort.ui.plotting.show()
h5filter.close()


"""
