import neo
import numpy as np
import active_vision.utils as avutils
from active_vision.fileio.lvdread import LVDReader
from fileio.dictionary_hdf5 import *
import h5py
import scipy
from elephant.signal_processing import butter as butterworth_filter
from conf.conf_files import projectdir, prepdir, rawdir, datadir, spkdir
import matplotlib.pyplot as plt


alldatasets = [
	[...]
	]

dataset = alldatasets[0]
data, subject, sess, rec, blk, pc = dataset

chans = np.array([0,1,2,3])

# Create HDF5 file
fn_hdf5 = "{dir}/Sorting_{subject}_{data}.hdf5".format(dir=datadir, subject=subject, data=data)


if not os.path.exists(fn_hdf5):
	H5 = dict()
else:
	# f_H5 = h5py.File(fn_hdf5, 'r')
	H5 = load_dict_from_hdf5(fn_hdf5)
	# f_H5.close()





sess_id = 'NEX_{sbj}_sess{sess}_rec{rec}_blk{blk}'.format(sbj=subject, sess=sess, rec=rec, blk=blk)
H5[sess_id] = {}
unit_dict = H5[sess_id]

chans = chans - 4

unit_IDs =[]
for i in range(6):
	chans = chans + 4
	if chans[0] >= 24: continue


	# filename from plexon NEX
	fn_nex = "{dir}/rawdata_matlab/rawdata_{sbj}_sess{sess}_rec{rec}_blk{blk}_pc{pc}_channels_{ch0}to{ch1}.nex".format(
		dir=datadir,
		sbj=subject, sess=sess, rec=rec, blk=blk, pc=pc, ch0=chans[0], ch1=chans[-1])


	print fn_nex

	for j, ch in enumerate(chans):
		if ch > 21: continue
		r = neo.io.NeuroExplorerIO(fn_nex)
		seg = r.read_segment()
		sts = [st for st in seg.spiketrains  if st.annotations['channel_index'] == j]

		segment = 0

		st_list =[]
		for s, st in enumerate(sts):
			unit_id = '{0:02d}{1:02d}{2:02d}'.format(ch, segment, s)
			unit_dict[unit_id]= st.magnitude

			print " - unit:", unit_id, 'added.'
			unit_IDs.append(unit_id)


H5['NEX_units_IDs'] = unit_IDs
H5['NEX_filter_sett'] = "Hpass of 500Hz, plexon LFilter"

print 'unit list:', unit_IDs

print "Savint to HDF5 ..."
# f_H5 = h5py.File(fn_hdf5, 'w+')
# save_dict_to_hdf5(H5,  hdf5file= f_H5)
# f_H5.close()

save_dict_to_hdf5(H5, filename=fn_hdf5)
