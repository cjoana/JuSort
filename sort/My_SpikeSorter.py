import neo
import numpy as np
import active_vision.utils as avutils
from active_vision.fileio.lvdread import LVDReader
import h5py
import scipy
from elephant.signal_processing import butter as butterworth_filter
from conf.conf_files import projectdir, prepdir, rawdir, datadir, spkdir
import matplotlib.pyplot as plt
# from scipy.cluster.vq import kmeans
from sklearn.decomposition import PCA
from scipy.cluster.vq import kmeans2 as KK
import hdbscan

import seaborn as sns
import sklearn.cluster as cluster
import time
sns.set_context('poster')
sns.set_color_codes()
# plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0, 'marker':','}
plot_kwds = {'alpha' : 0.25, 'linewidths':0, 'marker':',', 'lw':0, 's':3}



def plot_clusters(data, algorithm, args, kwds):
	start_time = time.time()
	labels = algorithm(*args, **kwds).fit_predict(data)
	end_time = time.time()
	palette = sns.color_palette('deep', np.unique(labels).max() + 1)
	colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
	plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
	# frame = plt.gca()
	# frame.axes.get_xaxis().set_visible(False)
	# frame.axes.get_yaxis().set_visible(False)

	label_types, unit_types = np.unique(labels, return_counts=True)

	plt.title('Clusters found by {}_{}'.format(str(algorithm.__name__), unit_types), fontsize=14)
	plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
	plt.show()
	plt.clf()

	return labels, colors

def align_waveforms(waveforms, window=[-1,2], NUM_POINTS= 10):
	from scipy.interpolate import interp1d
	"""
	:param spiketrain: spike train in (s)
	:param channel: mean channel where the neuron is detected
	:param fn_rawdata: path of the rawdata file lvp
	:param window: time-window that read from the raw data
	:param win_range:  time-window used when store the waveform (equal or smaller than "window")
	:param Fs: sampling-rate.

	:return: mean waveform (array) (uV), times since spike (win_range)  (ms) ,
			standard deviation of  mena, and CI up/down.
	"""
	## Waveform extraction:

	al_waveforms = []
	waveform_amplitudes = []
	for i, wv in enumerate(waveforms):

		## Linear interpolation
		xtimes = np.linspace(window[0], window[1], len(wv)*NUM_POINTS)
		times = np.linspace(window[0], window[1], len(wv))
		interp1 = interp1d(times, wv, kind='cubic')
		# interp1 = interp1d(times, wv, kind='linear')
		wv = interp1(xtimes)
		times = xtimes

		# ## Correct T_shift:
		t0 = int(np.where(times >= 0.000)[0][0])
		len10pc = NUM_POINTS * 10  #  +/- 0.5 ms
		dt = int(np.argmin(wv[t0-len10pc:t0+len10pc]) + (t0-len10pc) - t0)

		# print 't0, and dt:', t0, dt
		annex = [0] * np.abs(dt)
		if dt < 0:
			wv = np.hstack([annex, wv[:dt]])
		elif dt > 0:
			wv = np.hstack([wv[dt:], annex])

		al_waveforms.append(wv)
		waveform_amplitudes.append(wv[t0])


	return np.array(al_waveforms), np.array(waveform_amplitudes)

def runningMeanFast(x, N):
	return np.convolve(x, np.ones((N,))/N)[(N-1):]
def runningMeanFastReverse(x, N):
	x = x[::-1]
	return (np.convolve(x, np.ones((N,))/N)[(N-1):])[::-1]

"""" Spike sorting """
"""" Loading , filtering rawdata , waveform detection """

rawdir2 = "/datasets/..."
# rawdir =rawdir2

Filter_Data = True
Hpass = 250
Lpass = 7500
order = 4
Fs = 20000
filter_method = 'lfilter'


# chans = np.arange(0,12)
chans = np.array([2])


alldatasets = [
		["V1", 'SATSUKI', 20150911, 6, 4, 1],
	# 	["V1", 'SATSUKI', 20150911, 6, 2, 1],
	# 	["V1", 'SATSUKI', 20151127, 6, 2, 1],
	# ["V1", 'SATSUKI', 20151127, 6, 2, 1],
	# 	#
	# ["V1", 'SATSUKI', 20151110, 7, 2, 1],
	# 	# ["IT", 'SATSUKI', 20151110, 7, 2, 2],
	# 	# ["V1", 'SATSUKI', 20151110, 7, 5, 1],
	# 	# ["IT", 'SATSUKI', 20151110, 7, 5, 2],
	# 	# ["V1", 'HIME', 20141110, 3, 5, 1],
	# 	# ["IT", 'HIME', 20141110, 3, 5, 2],
	# 	# ["V1", 'HIME', 20141110, 3, 11, 1],
	# 	# ["IT", 'HIME', 20141110, 3, 11, 2],
	# 	# #
	# 	# ["V1", 'HIME', 20140909, 5, 2, 1],
	# 	# ["V1", 'HIME', 20140909, 5, 6, 1],
	# 	# ["V1", 'HIME', 20140909, 5, 9, 1],
	# 	# ["IT", 'HIME', 20140909, 5, 2, 2],
	# 	# ["IT", 'HIME', 20140909, 5, 6, 2],
	# 	# ["IT", 'HIME', 20140909, 5, 9, 2]
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
data = lvd_reader.get_data(channel=list(23 - chans), timerange=(blk_start, blk_end))[0]
if Filter_Data ==True :data = butterworth_filter(data, Hpass, Lpass, order=order, fs=Fs,filter_function=filter_method)
# data = lvd_reader.get_data(channel=23 - channel  ,  timerange=(t_start, t_end))[0]

times_rawdata = np.linspace(blk_start, blk_end, len(data), endpoint=True)

# Calculates the 'noise', 5 * std of filtered rawdata.
data_mean = np.mean(data)
data_std = np.std(data)
Noise = - data_std *5   # Warning: hard-coding (6* STD)


sk = 24
cutting = np.array(data < Noise, dtype=np.int)
kernel= np.array([1]*sk)
cut2 = np.convolve(  cutting, kernel) [:-sk+1]
cutting[cut2>1]=0


detections = np.argwhere(cutting==1)
detections = np.array([ int(x) for x in detections  ])
ISI = np.hstack([0,np.diff(detections)]) < 25

print 'ISI:', np.sum(ISI)

detections = detections[~ISI]



if len(detections) < 500:  # Warning: hard-coding min 500 threshold crossing
	print "Not enough waveforms detected,  --> no units in this channel"


# WARNING: hard-code (40 samples = 2 ms)
waveforms = np.array([ data[ int(arg) -20: int(arg) +40] for arg in detections])

print "lenght of waveform (samples)",  len(waveforms[0]), ", num events", len(waveforms)

# Align waveforms and Up-sample
waveforms, waveforms_amplitudes = align_waveforms(waveforms)

mean_wv = np.mean(waveforms,axis=0)




""" SEGMENTATION """

RMean = runningMeanFast(waveforms_amplitudes,200)
RMeanRev = runningMeanFastReverse(waveforms_amplitudes,200)
RMean_t = - data_std * 8
# discard = np.array(RMean > RMean_t , dtype=bool)
# diff = np.abs(np.hstack([0, np.diff(RMean)]))
diff = np.abs( RMean - RMeanRev)

def_points = np.array(diff >= 0.05,dtype=bool) #| discard
# def_points = np.array( diff >= 0.005 , dtype=int )
# kernel= np.array([1,1])
# def_points = np.convolve( def_points, kernel) [:-1]

global_st = times_rawdata[detections]

N_conv =200
segment = np.array(np.convolve(def_points, [1]*N_conv)[:-N_conv] >= 1, dtype=int)




diff_seg = np.hstack([np.diff(segment), 0])


seg_starts = global_st[np.argwhere(diff_seg <0 )]
seg_ends = global_st[np.argwhere(diff_seg >=1 )]
seg_starts = np.array( [int(x) for x in seg_starts])
seg_ends = np.array( [int(x) for x in seg_ends])

print np.sum( diff_seg == 1), np.sum( diff_seg == -1)
print seg_starts
print seg_ends

plt.plot(global_st, waveforms_amplitudes , 'k.', ms=5, alpha=0.2)
plt.plot(global_st[def_points], np.zeros_like(global_st[def_points]), 'b.', ms=20)
plt.plot(global_st, RMean, "r-")
plt.plot(global_st, RMeanRev, "g-")
plt.axhline(Noise)
plt.axhline(RMean_t)
for t in seg_starts : plt.axvline(t, c='g')
for t in seg_ends : plt.axvline(t, c='r')

plt.show()
plt.clf()





# seg_id =0
# print seg_starts[seg_id], seg_ends[seg_id]
# mask_seg1 = (global_st >= seg_starts[seg_id]) & (global_st <= seg_ends[seg_id])
# print mask_seg1, np.sum(mask_seg1)
# waveforms = waveforms[mask_seg1]
# waveforms_amplitudes = waveforms_amplitudes[mask_seg1]
# global_st = global_st[mask_seg1]
# RMean = RMean[mask_seg1]
# RMeanRev = RMeanRev[mask_seg1]
# def_points = def_points[mask_seg1]



""" Clustering """

# from scipy.cluster.vq import whiten
# pca_waveforms = whiten(waveforms)
# pca_waveforms = [ np.hstack([waveforms[x, 50:], np.argmax(waveforms[x, 79:]) - np.argmin(waveforms[x, 79:]) ]) for x in range(len(waveforms))]
pca_waveforms = waveforms[:,:]

pca = PCA(n_components=2, whiten=True, svd_solver="full")
pca.fit(pca_waveforms)
PCs = pca.transform(pca_waveforms).T
pc1 = PCs[0]
pc2 = PCs[1]
# pc3 = PCs[2]
WF = np.array([pc1, pc2]).T

# WF = PCs


plt.plot(mean_wv)
# plt.plot(PCs[0], lw=2)
# plt.plot(PCs[1], lw=2)
plt.axhline(data_mean, c='k')
plt.axhline(-Noise, c ='r')
plt.axhline(Noise, c='r')

plt.show()
plt.clf()

# plt.plot(pc1, pc2, 'k,')
# plt.show()
# plt.clf()
# centroids, codes = KK(WF, 2, iter=20)
# plt.plot(pc1[codes==0], pc2[codes==0], 'y,')
# plt.plot(pc1[codes==1], pc2[codes==1], 'g,')
# plt.show()
# plt.clf()
# plot_clusters(WF, cluster.DBSCAN, (), {'eps':0.025})
# plt.clf()

# codes, colors = plot_clusters(WF, cluster.DBSCAN, (), {'eps':0.025})
codes, colors = plot_clusters(WF, hdbscan.HDBSCAN, (), {'min_cluster_size':100, 'min_samples':6,
                                                        'allow_single_cluster':True, 'gen_min_span_tree':False,
                                                        'approx_min_span_tree':False})

# codes = codes
types_codes = list(set(codes))
types_colors = list(set(colors))
print types_codes, types_colors


# plt.plot( np.mean(waveforms[codes ==-1], axis=0), lw=1, c = types_colors[0] )
for i, color in enumerate(types_colors):
    plt.plot( np.mean(waveforms[codes == i], axis=0), lw=1, c = types_colors[i] )
plt.axhline(-Noise, c ='r')
plt.axhline(Noise, c='r')

plt.show()
plt.clf()

st_1 = global_st[codes==0]
wv_1 = waveforms_amplitudes[codes==0]
st_2 = global_st[codes ==1]
wv_2 = waveforms_amplitudes[codes==1]
st_0 = global_st[codes ==-1]
wv_0 = waveforms_amplitudes[codes==-1]




for i, color in enumerate(types_colors):
	print color
	plt.plot(global_st[codes == i], waveforms_amplitudes[codes == i], '.', c = types_colors[i], ms=2)

plt.show()
plt.clf()

for i, color in enumerate(types_colors):
	plt.plot(global_st[codes == i], waveforms_amplitudes[codes == i], '.', ms=2)
plt.axhline(Noise, c='r')

plt.show()
plt.clf()



# TODO: unitID =  XXYYZZ  --> XX=chan, YY=segment, ZZ=unit_num_in_chan_and_seg
# TODO : - segmentate data and then sort unit per segment.
