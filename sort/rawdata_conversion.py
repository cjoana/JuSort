import numpy as np
import active_vision.utils as avutils
from active_vision.fileio.lvdread import LVDReader
import h5py
import scipy
from elephant.signal_processing import butter as butterworth_filter
from conf.conf_files import projectdir, prepdir, rawdir, datadir, spkdir
import matplotlib.pyplot as plt

rawdir2 = "/datasets/"
# rawdir =rawdir2

Filter_Data = False
Hpass = 250
Lpass = 7500
order = 4
Fs = 20000

# chans = np.arange(0,24)
# chans = 3

# chans = np.arange(0,12)
chans = np.arange(12,24)


# chans = np.arange(0,4)
# chans = np.arange(4,8)
# chans = np.arange(8,12)
# chans = np.arange(12,16)
# chans = np.arange(16,20)
chans = np.arange(20,24)

alldatasets = [
		["V1",...],

	]

dataset = alldatasets[0]
data, subject, sess, rec,blk, pc =dataset

fn_raw = avutils.find_filenames(rawdir, subject, sess, rec, 'lvd', pc=pc)[0]
fn_features = "{dir}/hdf5_files/{sbj}/features_{sess}_rec{rec}_blk{blk}_{data}_{sbj}.hdf5".format(
	dir=datadir, sbj=subject, data=data, blk=blk, rec=rec, sess=sess)

f = h5py.File(fn_features, 'r')
blk_start = f["blk_start"][...]
blk_end = f["blk_end"][...]

print "file found, loading ..."
lvd_reader = LVDReader(fn_raw)
print 'blk_times', blk_start, blk_end
print lvd_reader.get_header()

print "Loading channels...", chans
# data = lvd_reader.get_data(channel=1)
data = lvd_reader.get_data(channel=list(23 - chans), timerange=(blk_start, blk_end))
if Filter_Data ==True :data = butterworth_filter(data, Hpass, Lpass, order=order, fs=Fs)
# data = lvd_reader.get_data(channel=23 - channel  ,  timerange=(t_start, t_end))[0]
nd = len(data)
print "Done.", nd, len(data[0]), data.size
print "max", np.max(data), np.min(data)


# data = (2**12-1) * data * 10**(-6) / 5

print data[0]

fn_bin = projectdir+"/bin_example.dat"
fn_mat = projectdir+"/rawdata_matlab/rawdata_{sbj}_sess{sess}_rec{rec}_blk{blk}_pc{pc}_channels_{ch0}to{ch1}".format(
	sbj=subject,sess=sess, rec=rec, blk=blk,pc=pc,ch0=chans[0], ch1=chans[-1])
if isinstance(chans, int): fn_mat = projectdir+"/rawdata_{sbj}_sess{sess}_rec{rec}_blk{blk}_pc{pc}_chan{chan}".format(
	sbj=subject,sess=sess, rec=rec, blk=blk,pc=pc, chan=chans)
if Filter_Data:
	fn_mat += "_filt"

# data.tofile(fn_bin)
scipy.io.savemat(fn_mat, mdict={'arr': data})
print "outfile:", fn_mat


#
#
# # with open(projectdir+"/bin_example.dat", "wb") as f:
# # 	f.write( data.tobytes())
# #
# #
# # bfile= open(projectdir+"/bin_example.dat", "rb")
# # for b in bfile.read():
# # 	print b
#
#
# with open(fn_bin, 'rb') as fd:
# 	x = np.fromfile(fd) # first 4 bytes (2 x uint16, big endian) give the length of the header in bytes.
# 	print x
# 	print len(x)
