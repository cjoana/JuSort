from JuSorter import *
import os
from scipy import stats
from fileio.dictionary_hdf5 import *

Hpass  = 500
Lpass = None
order = 4
Fs = 20000
filt_method = 'filtfilt'
# filt_method = 'lfilter'

segment_data = False

# channels = np.array([3])
channels = np.arange(24)

alldatasets = [ 	["V1", ... , 6, 4, 1],  	]

dataset = alldatasets[0]
data, subject, sess, rec, blk, pc = dataset

fn_raw = avutils.find_filenames(rawdir, subject, sess, rec, 'lvd', pc=pc)[0]
fn_features = "{dir}/hdf5_files/{sbj}/features_{sess}_rec{rec}_blk{blk}_{data}_{sbj}.hdf5".format(
	dir=datadir, sbj=subject, data=data, blk=blk, rec=rec, sess=sess)

f = h5py.File(fn_features, 'r')
blk_start = f["blk_start"][...]
blk_end = f["blk_end"][...]

# Create HDF5 file
# fn_hdf5 = "{dir}/hdf5_files/ReSorting_{subject}_{data}.hdf5".format(dir=datadir, subject=subject, data=data)
#
# if not os.path.exists(fn_hdf5):
# 	H5 = dict()
# else:
# 	# f_H5 = h5py.File(fn_hdf5, 'r')
# 	H5 = load_dict_from_hdf5(fn_hdf5)
# 	# f_H5.close()
#
# sess_id = 'JuS_{sbj}_sess{sess}_rec{rec}_blk{blk}'.format(sbj=subject, sess=sess, rec=rec, blk=blk)
# H5[sess_id] = {}
# unit_dict = H5[sess_id]


unit_IDs =[]
for channel in channels:
	channel = np.array([channel])

	# Get filtered raw_data
	rawdat, times_rawdat = get_filtered_data(fn_raw, channel, blk_start, blk_end, Hpass=Hpass, Lpass=Lpass, order=order, filter_m=filt_method)

	# Detect waveforms, and Aliginig+UpSampling
	threshold = -0.05#- np.std(rawdat, axis=0) * 6
	waveforms, detect_array = waveform_detection(rawdat, threshold= threshold, post_deadsamples=25)
	if len(waveforms) < 10 :
		print "skipping channel, not enough events."
		continue
	waveforms, waveforms_amplitude = align_waveforms(waveforms, window=[-1, 2], NUM_POINTS=10)
	wv_times = np.linspace(-1,2, len(waveforms[0]))

	# Create a global spiketrain
	global_st = times_rawdat[detect_array]

	# Detect timings of segments in rawdata
	if segment_data:
		seg_starts, seg_ends, rm1, rm2 = segmentatition_times(global_st, waveforms_amplitude, rm_th=0.05 , N_conv=50)
		segments = [ [seg_starts[i], seg_ends[i]] for i in range(len(seg_starts))]
	else:
		seg_starts, seg_ends, rm1, rm2 = segmentatition_times(global_st, waveforms_amplitude, rm_th=0.05, N_conv=50)
		segments = [ [blk_start, blk_end] ]


	seg_num = 0
	units_sts = []
	units_waveforms = []
	units_waveforms_amplitude = []
	units_ID = []
	for times in segments:

		seg_str = times[0]
		seg_end = times[1]
		seg_dur = seg_end - seg_str

		if seg_dur <= 200: # Warning!: Hard coding in min seg interval

			continue

		print ' Sorting segment', seg_num, 'of', len(segments), "[{},{}]".format(seg_str,seg_end)


		# mask_seg = (global_st >= seg_str) & (global_st <= seg_end)
		# mask_seg = np.ones_like(waveforms, dtype=bool)
		seg_waveforms = waveforms
		seg_waveforms_amplitude = waveforms_amplitude
		seg_st = global_st


		# Cluster into SUAs:
		# labels,  PCs = clustering(seg_waveforms, 'ScanKmeans')
		labels, PCs = clustering(seg_waveforms)
		codes = labels.copy()

		# v , cod_ind, types_codes = np.unique(codes, return_index=True, return_counts= True)
		types_codes = list(set(codes))
		print types_codes


		# Merging same-shaped SUAs
		MWs = []
		for i, code in enumerate(types_codes):
			if code <0: continue
			mw = np.mean(seg_waveforms[codes == types_codes[i], int(len(seg_waveforms[0])/5): ], axis=0)
			MWs.append(mw)


		skip_merged = []
		new_codes = codes.copy()
		for i, im in enumerate(MWs):
			if i in skip_merged: continue
			cnt = 0
			for j, jm in enumerate(MWs):
				if i == j:	continue
				# if cnt ==1 : continue

				slope, intercept, r_value, p_value, std_err = stats.linregress(im, jm)
				print i, j, slope, r_value

				if slope <=1.0 and slope >= 0.75 and r_value <=1.0 and r_value >= 0.95:
					print "mergin unit", i , j
					new_codes[new_codes == types_codes[j]] = types_codes[i]
					skip_merged.append(j)
					cnt = 1

		print 'lengths:', len(codes), len(new_codes)

		codes = new_codes
		types_codes = np.unique(codes)


		unit_num = 0
		for i, ct in enumerate(types_codes):
			if np.sum(codes == ct) <= 500: continue   # Warning! Hard coding in min of spikes per unit
			if ct < 0: continue
			units_sts.append(seg_st[codes == ct ])
			units_waveforms.append(seg_waveforms[codes == ct])
			units_waveforms_amplitude.append(seg_waveforms_amplitude[codes == ct])
			units_ID.append('{0:02d}{1:02d}{2:02d}'.format(int(channel), seg_num, unit_num))
			unit_num += 1

		seg_num += 1




	colors = np.array([palette[x] if x >= 0 else 'k' for x in codes])
	types_colors =  palette[: len(types_codes)] # np.unique(colors)
	print types_colors




	""" Exporting SUA to HDF5 """

	sts = [ global_st[codes == code] for code in types_codes ]
	# for s, st in enumerate(sts):
	# 	segment = 0
	# 	unit_id = '{0:02d}{1:02d}{2:02d}'.format(int(channel), segment, s)
	# 	unit_dict[unit_id] = st
	#
	# 	print " - unit:", unit_id, 'added.'
	# 	unit_IDs.append(unit_id)








	""" Plotting results """
	num_plot_types = 4  # types of plots: 4:  waveform, ISIH, PSTH_sac, PSTH_fix,  two rasterplots
	num_row, num_col = [2, 2]
	n_plots = num_col * num_row
	fig = plt.figure(figsize=(6 * num_row, 5 * num_col))
	fig.subplots_adjust(left=0.05, right=0.96, top=0.90, bottom=0.22, wspace=0.32, hspace=0.4)
	num_plot_types = num_plot_types - 1  # iteration counting like 0,1,2,3
	x_plot = 0



	x_plot += 1
	""" Plot rawdata"""
	ax = fig.add_subplot(num_col, num_row, x_plot)
	for i, color in enumerate(types_colors):
		ax.plot(global_st[codes == types_codes[i]], waveforms_amplitude[codes == types_codes[i]], 'o', c=types_colors[i], ms=2)
	for t in range(len(seg_starts)):
		ax.axvline(seg_starts[t], c='g')
		ax.axvline(seg_ends[t], c='r')
	ax.plot(global_st, rm1, 'k', alpha=0.9, lw=1)
	ax.plot(global_st, rm2, 'k', alpha=0.7, lw=1)
	ax.set_xlabel('Spiketimes (s)')
	ax.set_ylabel('waveform amplitude')


	x_plot += 1
	""" Plot PCA analysis & Clustering"""
	ax = fig.add_subplot(num_col, num_row, x_plot)
	ax.scatter(PCs[0].T, PCs[1].T, c=colors, **plot_kwds)
	ax.set_xlabel('PC1')
	ax.set_ylabel('PC2')




	x_plot += 1
	""" Plot SUA mean waveform"""
	ax = fig.add_subplot(num_col, num_row, x_plot)

	MWs =[]
	for i, code in enumerate(types_codes):
		mw = np.mean(waveforms[codes == code], axis=0)
		std = np.std(waveforms[codes == code], axis=0)
		ax.plot(wv_times, mw, lw=1, c=types_colors[i],
				label='unit {} ({})'.format(i, np.sum(codes == code)))
		ax.fill_between(wv_times, mw - std, mw + std, color=types_colors[i], alpha=0.3, edgecolor=types_colors[i])
		# ax.fill_between(mw - std, mw + std, color="gray", alpha=0.3, edgecolor='k')
		ax.legend()
		MWs.append(mw)
	ax.axhline(-threshold, c='r')
	ax.axhline(threshold, c='r')
	ax.set_xlabel('Time (ms)')
	ax.set_ylabel('Voltage (uV)')




	# x_plot += 1
	# """ text  """
	# ax = fig.add_subplot(num_col, num_row, x_plot)
	# LI = []
	# yt = 0.8
	# ydt = 0.2
	# for i, im in enumerate(MWs):
	# 	li=[]
	# 	for j, jm in enumerate(MWs):
	# 		slope, intercept, r_value, p_value, std_err = stats.linregress(im, jm)
	# 		li.append( "{0:.2} - {1:.2}".format(slope, r_value)  )
	# 	LI.append(li)
	#
	# 	ax.text(-0.1, yt - i*ydt, li )


	x_plot += 1
	""" Plot PCA analysis & Clustering"""
	ax = fig.add_subplot(num_col, num_row, x_plot)
	ax.scatter(PCs[2].T, PCs[1].T, c=colors, **plot_kwds)
	ax.set_xlabel('PC3')
	ax.set_ylabel('PC2')

	# ax.axes(None)






	"""  SAVING 	PLOT 	"""

	# ####
	# figure_dir = "{datadir}/figures/waveforms/sorting/JuSorter".format( datadir=projectdir)
	#
	# if not os.path.exists(figure_dir):
	# 	os.makedirs(figure_dir)
	#
	# fn = "{figdir}/sess{sess}_rec{rec}_blk{blk}_chan{chan}_{subject}.png".format(figdir=figure_dir, datadir=projectdir,
	# 																			 subject=subject, sess=sess, rec=rec, blk=blk, chan=channel)

	## FILTER SETTINGS
	figure_dir = "{datadir}/figures/waveforms/sorting/JuSorter/filter_settings/".format(datadir=projectdir)

	if not os.path.exists(figure_dir):
		os.makedirs(figure_dir)

	fn = "{figdir}/sess{sess}_rec{rec}_blk{blk}_chan{chan}_{subject}_H{H}_L{L}_{F}_o{o}.png".format(figdir=figure_dir, datadir=projectdir,
																				 subject=subject, sess=sess, rec=rec,
																				 blk=blk, chan=channel,H=Hpass, L=Lpass, F=filt_method, o=order )



	plt.tight_layout()
	plt.savefig(fn)
	# plt.savefig("{figdir}/a_test.png".format(figdir=figure_dir))

	plt.cla()
	plt.clf()
	plt.close()





# # Save SUA file
# figure_dir = "{datadir}/figures/waveforms/sorting/JuSorter".format(
# 	datadir=projectdir)
#
# if not os.path.exists(figure_dir):
# 	os.makedirs(figure_dir)


# H5['JuS_units_IDs'] = unit_IDs
#
# print 'unit list:', unit_IDs
#
# print "Savint to HDF5 ..."



# save_dict_to_hdf5(H5, filename=fn_hdf5)

