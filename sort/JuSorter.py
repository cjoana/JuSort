import neo
import numpy as np
import sys
sys.path.append("/home/joana/toolbox")
import active_vision.utils as avutils
from active_vision.fileio.lvdread import LVDReader
import h5py
import scipy
from elephant.signal_processing import butter as butterworth_filter
from conf.conf_files import projectdir, prepdir, rawdir, datadir, spkdir
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from scipy.cluster.vq import kmeans2 as KK
import hdbscan
import seaborn as sns
import sklearn.cluster as cluster
from scipy import stats
import time
sns.set_context('poster')
sns.set_color_codes()
# plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0, 'marker':','}
plot_kwds = {'alpha' : 0.25, 'linewidths':0, 'marker':'o', 'lw':0, 's':8}
palette = np.array(['b','r','g','yellow','orange','purple','c','m','k'])


def datasets_from_odml_segmentation(dir, subject="SATSUKI"):
	import glob

	dlist = glob.glob(dir + "/*.odml")
	sep = [dir + "/", "_rec", "_blk", "_"]

	datasets = []
	for d in dlist:

		d = d.split(sep[0])[-1]
		sess = int(d.split(sep[1])[0])
		d = d.split(sep[1])[-1]
		rec = int(d.split(sep[2])[0])
		d = d.split(sep[2])[-1]
		blk = int(d.split(sep[3])[0])
		data = d.split(sep[3])[1]

		if data == "V1":
			pc = 1
		elif data == "IT":
			pc = 2
		else:
			print "  -->", data, subject, sess, rec, blk
			raise ("data must be V1 or IT")

		dset = [data, subject, sess, rec, blk, pc]
		datasets.append(dset)

		datasets.sort(key=lambda x: x[2])

	return datasets


def get_filtered_data(fn_raw, channel, blk_start, blk_end, Hpass=500, Lpass=None, order=4, Fs=20000,
					  filter_m='filtfilt'):
	if Hpass or Lpass:
		Filter_Data = True
	else:
		Filter_Data = False

	print "Loading rawdata ..."
	lvd_reader = LVDReader(fn_raw)

	print "Loading channels...", channel
	data = lvd_reader.get_data(channel=list(23 - channel), timerange=(blk_start, blk_end))[0]
	if Filter_Data == True: data = butterworth_filter(data, Hpass, Lpass, order=order, fs=Fs,
													  filter_function=filter_m)

	times = np.linspace(blk_start, blk_end, len(data), endpoint=True)

	return data, times


def waveform_detection(data, threshold=- 0.1, post_deadsamples=25):
	cutting = np.array(data < threshold, dtype=np.int)
	kernel = np.array([1] * post_deadsamples)
	cut2 = np.convolve(cutting, kernel)[:-post_deadsamples + 1]
	cutting[cut2 > 1] = 0

	# Boolean array with samples containing spikes
	detections = np.argwhere(cutting == 1)
	detections = np.array([int(x) for x in detections])

	# Avoiding overlaping spikes
	ISI = np.hstack([0, np.diff(detections)]) < post_deadsamples  # 20 samples = 1 ms
	detections = detections[~ISI]

	if len(detections) < 500:  # Warning: hard-coding min 500 threshold crossing
		print "Not enough waveforms detected,  --> no units in this channel"
		return [], []

	# WARNING: hard-code (40 samples = 2 ms)
	waveforms = np.array([data[int(arg) - 20: int(arg) + 40] for arg in detections])
	print "  - Num detec. events:", len(waveforms)

	return waveforms, detections


def align_waveforms(waveforms, window=[-1, 2], NUM_POINTS=10):
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

	al_waveforms = []
	waveform_amplitudes = []
	for i, wv in enumerate(waveforms):

		## Linear interpolation
		xtimes = np.linspace(window[0], window[1], len(wv) * NUM_POINTS)
		times = np.linspace(window[0], window[1], len(wv))
		interp1 = interp1d(times, wv, kind='cubic')
		# interp1 = interp1d(times, wv, kind='linear')
		wv = interp1(xtimes)
		times = xtimes

		## Aligning at minimum (valley)
		t0 = int(np.where(times >= 0.000)[0][0])
		len10pc = NUM_POINTS * 10  # +/- 0.5 ms
		dt = int(np.argmin(wv[t0 - len10pc:t0 + len10pc]) + (t0 - len10pc) - t0)
		annex = [0] * np.abs(dt)
		if dt < 0:
			wv = np.hstack([annex, wv[:dt]])
		elif dt > 0:
			wv = np.hstack([wv[dt:], annex])

		al_waveforms.append(wv)
		waveform_amplitudes.append(wv[t0])

	return np.array(al_waveforms), np.array(waveform_amplitudes)


def runningMeanFast(x, N):
	return np.convolve(x, np.ones(N) / N)[(N - 1):]


def runningMeanFastReverse(x, N):
	x = x[::-1]
	# return (np.convolve(x, np.ones((N,)) / N)[(N - 1):])[::-1]
	return (np.convolve(x, np.ones(N) / N)[(N - 1):])[::-1]


def segmentatition_times(st, waveforms_amplitudes, rm_th=0.01, N_conv=50):
	RMean = runningMeanFast(waveforms_amplitudes, 200)
	RMeanRev = runningMeanFastReverse(waveforms_amplitudes, 200)

	diff = np.abs(RMean - RMeanRev)

	def_points = np.array(diff >= rm_th, dtype=bool)
	segment = np.array(np.convolve(def_points, [1] * N_conv)[:-N_conv] >= 1, dtype=int)

	diff_seg = np.hstack([np.diff(segment), 0])

	seg_starts = st[np.argwhere(diff_seg < 0)]
	seg_ends = st[np.argwhere(diff_seg >= 1)]
	seg_starts = np.array([int(x) for x in seg_starts])
	seg_ends = np.array([int(x) for x in seg_ends])

	if len(seg_starts) != len(seg_ends):
		print '!! -- WARNING: len(seg_starts) ({}) differs from len(seg_ends) ({})'.format(len(seg_starts),
																						   len(seg_ends))

	return seg_starts, seg_ends, RMean, RMeanRev


def clustering(waveforms, algorithm=hdbscan.HDBSCAN, args=(),kwds={'cluster_selection_method': 'leaf', 'min_cluster_size': 50,'min_samples':1 }):
	"""Alternative method:"""
	## cluster.DBSCAN, (), {'eps':0.025}



	pca_waveforms = np.copy(waveforms)

	pca = PCA(n_components=5, whiten=False, svd_solver="full")
	pca.fit(pca_waveforms)
	PCs = pca.transform(pca_waveforms).T
	pc1 = PCs[0]
	pc2 = PCs[1]
	pc3 = PCs[2]

	# PCA Matrix for clustering
	# WF = np.array([pc1, pc2]).T
	WF = np.array([pc1, pc2, pc3]).T



	if algorithm == 'ScanKmeans':
		# labels = hdbscan.HDBSCAN(*args, **kwds).fit_predict(WF)
		n_clust = 100
		kmeans = cluster.KMeans(n_clusters=n_clust)
		labels = kmeans.fit_predict(WF)
	else:
		labels = algorithm(*args, **kwds).fit_predict(WF)


	return labels, PCs


def merge_leaf(waveforms, codes):
	types_codes = np.unique(codes)

	# Merging same-shaped SUAs
	MWs = []
	pos_codes = types_codes[types_codes >= 0]
	for i, code in enumerate(pos_codes):
		# if code <0: continue
		mw = np.mean(waveforms[codes == code], axis=0)
		MWs.append(mw)

	num_merges = 100
	while num_merges > 0:
		num_merges = 0
		new_codes = codes.copy()
		skip_merged = []
		print "initial num cluster", len(np.unique(codes)), np.unique(codes)
		for i, im in enumerate(MWs):
			if i in skip_merged: continue
			for j, jm in enumerate(MWs):
				if i >= j:    continue

				slope, intercept, r_value, p_value, std_err = stats.linregress(im, jm)
				# print i, j, slope, r_value

				if r_value  >= 0.9:
					print "mergin unit", i, j
					new_codes[new_codes == pos_codes[j]] = pos_codes[i]
					skip_merged.append(j)
					num_merges += 1

		print "TOTAL MERGES", num_merges

		codes = new_codes
		types_codes = np.unique(codes)
		pos_codes = types_codes[types_codes >= 0]

		# Recalculating mean waveforms
		MWs = []
		for i, code in enumerate(pos_codes):
			mw = np.mean(waveforms[codes == code], axis=0)
			MWs.append(mw)

	# codes = new_codes
	types_codes, types_counts = np.unique(codes[codes >=0], return_counts=True)

	for i, c in enumerate(types_codes):
		if types_counts[i] < 500:
			codes[codes==c]=-1

	types_codes, types_counts = np.unique(codes[codes >= 0], return_counts=True)

	# Sort 0,1,2,.. by lenght (0-> longest)
	sorted_type_codes = types_codes[np.argsort(types_counts)[::-1]]
	rescale = len(sorted_type_codes) + 10
	for i, code in enumerate(sorted_type_codes):
		codes[codes == code] = i + rescale
	codes[codes >= 0] += - rescale


	types_codes= np.unique(codes)

	return codes, types_codes

if __name__ == "__main__":



	channel = np.array([3])

	alldatasets = [ 		["V1",...]  	]

	dataset = alldatasets[0]
	data, subject, sess, rec, blk, pc = dataset

	fn_raw = avutils.find_filenames(rawdir, subject, sess, rec, 'lvd', pc=pc)[0]
	fn_features = "{dir}/hdf5_files/{sbj}/features_{sess}_rec{rec}_blk{blk}_{data}_{sbj}.hdf5".format(
		dir=datadir, sbj=subject, data=data, blk=blk, rec=rec, sess=sess)

	f = h5py.File(fn_features, 'r')
	blk_start = f["blk_start"][...]
	blk_end = f["blk_end"][...]

	# Get filtered raw_data
	rawdat, times_rawdat = get_filtered_data(fn_raw, channel, blk_start, blk_end, filter_m='lfilter')

	# Detect waveforms, and Aliginig+UpSampling
	threshold = np.std(rawdat) * 6
	waveforms, detect_array = waveform_detection(rawdat, threshold=-threshold, post_deadsamples=25)
	waveforms, waveforms_amplitude = align_waveforms(waveforms, window=[-1, 2], NUM_POINTS=10)

	# Create a global spiketrain
	global_st = times_rawdat[detect_array]

	# Detect timings of segments in rawdata
	seg_starts, seg_ends, RMean, RMeanRev = segmentatition_times(global_st, waveforms_amplitude, rm_th=0.05, N_conv=200)



	# Select segment
	# seg_id =1
	# mask_seg1 = (global_st >= seg_starts[seg_id]) & (global_st <= seg_ends[seg_id])
	# waveforms = waveforms[mask_seg1]
	# waveforms_amplitudes = waveforms_amplitudes[mask_seg1]
	# global_st = global_st[mask_seg1]


	# Cluster into SUAs:
	labels, colors,PCs = clustering(waveforms, 'Agglo')




	# Plotting results
	codes = labels.copy()

	types_codes = list(set(codes))
	types_colors = list(set(colors))
	print types_codes, types_colors





	for i, color in enumerate(types_colors):
		plt.plot(global_st[codes == i], waveforms_amplitude[codes == i], '.', c=types_colors[i], ms=2)
	for t in range(len(seg_starts)):
		plt.axvline(seg_starts[t], c='g')
		plt.axvline(seg_ends[t], c='r')
	plt.show()
	plt.clf()



	plt.scatter(PCs[0].T, PCs[1].T, c=colors, **plot_kwds)
	plt.show()
	plt.clf()




	# plt.plot( np.mean(waveforms[codes ==-1], axis=0), lw=1, c = types_colors[0] )
	for i, code in enumerate(types_codes):
		plt.plot(np.mean(waveforms[codes == code], axis=0), lw=1, c=types_colors[i])
	plt.axhline(-threshold, c='r')
	plt.axhline(threshold, c='r')

	plt.show()
	plt.clf()


