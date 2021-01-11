import numpy as np
import scipy.stats
import scipy.signal
import h5py
import matplotlib.pyplot as plt


def get_fdr_threshold(ps, q):
    pval_thresholds = np.linspace(q / len(ps), q, len(ps))
    idxs_rejected_pval = np.where(np.sort(ps) < pval_thresholds)[0]
    return 0 if len(idxs_rejected_pval) == 0 else pval_thresholds[idxs_rejected_pval.max()]


def segment_successive_occurrences(data, value):
    # detect the edges of successive occurrences of `value` in `data`
    data_bin = (data == value).astype(int)
    idx_inc = np.where(np.diff(data_bin) == 1)[0]
    idx_dec = np.where(np.diff(data_bin) == -1)[0]

    # special cases where the edges are too few and so the following boundary treatment doesn't work
    if len(idx_dec) == 0 and len(idx_inc) == 0:
        return None
    elif len(idx_dec) == 1 and len(idx_inc) == 0:
        idx_inc = np.array([-1,])
    elif idx_dec.size == 0 and idx_inc.size == 1:
        idx_dec = np.array([len(data) - 1,])

    # treatment of the boundaries in order to assure that idx_inc and idx_dec are of the same size and idx_inc[i] always
    # precedes idx_dec[i] for any i
    if idx_dec[0] < idx_inc[0]:
        idx_inc = np.hstack(([-1,], idx_inc))
    if idx_dec[-1] < idx_inc[-1]:
        idx_dec = np.hstack((idx_dec, [len(data) - 1,]))
    return np.array(zip(idx_inc + 1, idx_dec + 1))


def label_thresholded_segments(data, threshold):
    condition = data >= threshold
    supth_seg_edges = segment_successive_occurrences(condition, True)
    subth_seg_edges = segment_successive_occurrences(condition, False)

    # identify segments of stable and unstable rate, and assign segment IDs
    # segment ID 0: when no subthreshold segments are identified, ID 0 is assigned to the whole episode
    # segment ID 1, 2, 3, ...: suprathreshold segments with the longest duration, the 2nd longest, and so on
    # segment ID -1, -2, -3, ...: subthreshold segments with the longest duration, the 2nd longest, and so on
    seg_edges = {}
    if np.all(condition):
        seg_edges[0] = [0, len(data)]
    elif np.all(np.logical_not(condition)):
        seg_edges[-1] = [0, len(data)]
    else:
        # assign negative segment IDs to unstable rate segments
        segID = -1
        for i_seg in np.argsort([x[1]-x[0] for x in subth_seg_edges])[::-1]:
            seg_edges[segID] = subth_seg_edges[i_seg]
            segID -= 1
        # assign positive segment IDs to unstable rate segments
        segID = 1
        for i_seg in np.argsort([x[1]-x[0] for x in supth_seg_edges])[::-1]:
            seg_edges[segID] = supth_seg_edges[i_seg]
            segID += 1
    return seg_edges


def identify_maxamp_seg(seg_time_ranges, spike_times, spike_amplitudes, fr_threshold=None, seg_frs=None):
    if fr_threshold and seg_frs is None:
        raise ValueError("When fr_threshold is given, seg_frs must also be given.")

    if len(seg_time_ranges) == 1:
        segID_maxamp = seg_time_ranges.keys()[0]
    else:
        seg_amps = []
        seg_ids = []
        for segID, seg_time_range in seg_time_ranges.items():
            if segID < 0:
                continue
            if fr_threshold is not None and seg_frs[segID] < fr_threshold:
                continue
            t_ini, t_fin = seg_time_range
            amps = np.abs(spike_amplitudes[(t_ini < spike_times) & (spike_times < t_fin)])
            seg_amps.append(np.mean(amps))
            seg_ids.append(segID)
        if len(seg_amps) > 0:
            segID_maxamp = seg_ids[np.argmax(seg_amps)]
        else:
            segID_maxamp = None

    return segID_maxamp


def extend_segment_forward(pivot_seg_time_range, seg_time_ranges, pvals, pval_times, pval_threshold, fr_threshold=None, seg_frs=None):
    def find_next_segment(pivot_seg_time_range, seg_time_ranges, fr_threshold=None, seg_frs=None):
        preseg_end_time = pivot_seg_time_range[1]
        seg_begin_times = {}
        seg_end_times = {}
        for seg_id, seg_time_range in seg_time_ranges.items():
            if seg_id < 0:
                continue
            if fr_threshold is not None and seg_frs[seg_id] < fr_threshold:
                continue
            if seg_time_range[0] > preseg_end_time:
                seg_begin_times[seg_id] = seg_time_range[0]
                seg_end_times[seg_id] = seg_time_range[1]
        if len(seg_begin_times) == 0:
            return None
        else:
            return seg_begin_times.keys()[np.argmin(seg_begin_times.values())]

    def connect_segments(pivot_seg_time_range, seg_time_ranges, pvals, pval_times, pval_threshold, fr_threshold=None, seg_frs=None):
        postseg_id = find_next_segment(pivot_seg_time_range, seg_time_ranges, fr_threshold, seg_frs)
        if postseg_id is None:
            return pivot_seg_time_range
        preseg_end_time = pivot_seg_time_range[1]
        postseg_begin_time = seg_time_ranges[postseg_id][0]
        mask = (preseg_end_time < pval_times) & (pval_times < postseg_begin_time)
        if np.all(pvals[mask] >= pval_threshold):
            pivot_seg_time_range_new = [pivot_seg_time_range[0], seg_time_ranges[postseg_id][1]]
            return connect_segments(pivot_seg_time_range_new, seg_time_ranges, pvals, pval_times, pval_threshold)
        else:
            return pivot_seg_time_range

    def extend_edge(pivot_seg_time_range, seg_time_ranges, pvals, pval_times, pval_threshold):
        seg_begin_time, seg_end_time = pivot_seg_time_range
        mask = seg_end_time < pval_times
        postseg_id = find_next_segment(pivot_seg_time_range, seg_time_ranges)
        if postseg_id:
            mask = mask & (pval_times < seg_time_ranges[postseg_id][0])
        pvals_edge = pvals[mask]
        pval_times_edge = pval_times[mask]
        if len(pvals_edge) == 0:
            pass
        elif np.all(pvals_edge > pval_threshold):
            seg_end_time = pval_times_edge.max()
        else:
            seg_end_time = pval_times_edge[np.argmin(pvals_edge)]
        return [seg_begin_time, seg_end_time]

    if fr_threshold and seg_frs is None:
        raise ValueError("When fr_threshold is given, seg_frs must also be given.")

    stable_seg_time_range = connect_segments(pivot_seg_time_range, seg_time_ranges, pvals, pval_times, pval_threshold, fr_threshold, seg_frs)
    stable_seg_time_range = extend_edge(stable_seg_time_range, seg_time_ranges, pvals, pval_times, pval_threshold)

    return stable_seg_time_range


def extend_segment_backward(pivot_seg_time_range, seg_time_ranges, pvals, pval_times, pval_threshold, fr_threshold=None, seg_frs=None):
    pivot_seg_time_range_rev = [-pivot_seg_time_range[1], -pivot_seg_time_range[0]]
    seg_time_ranges_rev = {}
    for seg_id, seg_time_range in seg_time_ranges.items():
        seg_time_ranges_rev[seg_id] = [-seg_time_range[1], -seg_time_range[0]]
    pval_times_rev = -pval_times
    pivot_seg_time_range_rev_new = extend_segment_forward(pivot_seg_time_range_rev, seg_time_ranges_rev,
                                            pvals, pval_times_rev, pval_threshold, fr_threshold, seg_frs)
    return [-pivot_seg_time_range_rev_new[1], -pivot_seg_time_range_rev_new[0]]


def find_stable_segment(spike_times, spike_amplitudes, trial_time_ranges, params):
    bin_sizes = {'tfr': params["TrialBinSize"], 'amp': params["SpikeBinSize"]}
    bin_steps = {'tfr': params["TrialBinStep"], 'amp': params["SpikeBinStep"]}
    idx_trial_ini = np.where(spike_times[0] < trial_time_ranges[:, 0])[0][0]
    idx_trial_fin = np.where(trial_time_ranges[:, 1] < spike_times[-1])[0][-1] + 1
    bin_edges = {
        'tfr': np.arange(idx_trial_ini, idx_trial_fin - 2*bin_sizes['tfr'] + 1, bin_steps['tfr']),
        'amp': np.arange(0, len(spike_times) - 2*bin_sizes['amp'], bin_steps['amp'])
    }
    fdr_q = params["FDR-q"]

    trial_spike_times = []
    trial_spike_amplitudes = []
    for i_trial, (t_ini, t_fin) in enumerate(trial_time_ranges):
        mask_trial = (t_ini <= spike_times) & (spike_times < t_fin)
        trial_spike_times.append(spike_times[mask_trial])
        trial_spike_amplitudes.append(spike_amplitudes[mask_trial])
    trial_firing_rates = [len(x) / (y[1] - y[0]) for x, y in zip(trial_spike_times, trial_time_ranges)]

    pvals = {}
    pval_times = {}
    pval_threshold = {}
    seg_time_ranges = {}

    # First step: apply trial firing rate (TFR) based segmentation
    # --- compute p-values of TFR stability in a sliding window manner
    bin_size = bin_sizes['tfr']
    pvals['tfr'] = np.empty(len(bin_edges['tfr']))
    pval_times['tfr'] = np.empty_like(pvals['tfr'])
    for i, idx_ini in enumerate(bin_edges['tfr']):
        fr_pre = trial_firing_rates[idx_ini : idx_ini+bin_size]
        fr_post = trial_firing_rates[idx_ini+bin_size : idx_ini+2*bin_size]
        _, pvals['tfr'][i] = scipy.stats.ks_2samp(fr_pre, fr_post)
        pval_times['tfr'][i] = (trial_time_ranges[idx_ini+bin_size-1][1] + trial_time_ranges[idx_ini+bin_size][0]) / 2

    # --- get the p-value threshold with Benjamini-Hochberg method of FDR
    pval_threshold['tfr'] = get_fdr_threshold(pvals['tfr'], fdr_q)

    # --- segment the p-values by the threshold and obtain the segment edge times
    seg_edges = label_thresholded_segments(pvals['tfr'], pval_threshold['tfr'])
    seg_time_ranges['tfr'] = {k: [pval_times['tfr'][v[0]], pval_times['tfr'][v[-1]-1]] for k, v in seg_edges.items()}

    # Second step: compute p-values of spike amplitude stability
    bin_size = bin_sizes['amp']
    pvals['amp'] = np.empty(len(bin_edges['amp']))
    pval_times['amp'] = np.empty_like(pvals['amp'])
    for i, idx_ini in enumerate(bin_edges['amp']):
        amps_pre = spike_amplitudes[idx_ini : idx_ini+bin_size]
        amps_post = spike_amplitudes[idx_ini+bin_size : idx_ini+2*bin_size]
        _, pvals['amp'][i] = scipy.stats.ks_2samp(amps_pre, amps_post)
        pval_times['amp'][i] = (spike_times[idx_ini + bin_size - 1] + spike_times[idx_ini + bin_size]) / 2
    pval_threshold['amp'] = get_fdr_threshold(pvals['amp'], fdr_q)
    seg_edges = label_thresholded_segments(pvals['amp'], pval_threshold['amp'])
    seg_time_ranges['amp'] = {k: [pval_times['amp'][v[0]], pval_times['amp'][v[-1]-1]] for k, v in seg_edges.items()}

    # Third step: connect stable segments based on spike amplitude information
    # --- identify the segment with the maximum mean spike amplitude
    seg_frs = {}
    for segID, seg_time_range in seg_time_ranges['tfr'].items():
        t_ini, t_fin = seg_time_range
        times = spike_times[(t_ini < spike_times) & (spike_times < t_fin)]
        seg_frs[segID] = 0.0 if len(times) == 0 else len(times) / (times[-1] - times[0])
    segID_maxamp = identify_maxamp_seg(seg_time_ranges['tfr'], spike_times, spike_amplitudes, fr_threshold=0.5, seg_frs=seg_frs)

    # --- connect segments starting from the identified segment
    # seg_frs = {}
    if segID_maxamp is None:
        stable_seg_time_range = None
    else:
        stable_seg_time_range = seg_time_ranges['tfr'][segID_maxamp]
        stable_seg_time_range = extend_segment_forward(stable_seg_time_range, seg_time_ranges['tfr'],
                                                pvals['amp'], pval_times['amp'], pval_threshold['amp'],
                                                fr_threshold=0.5, seg_frs=seg_frs)
        stable_seg_time_range = extend_segment_backward(stable_seg_time_range, seg_time_ranges['tfr'],
                                                pvals['amp'], pval_times['amp'], pval_threshold['amp'],
                                                fr_threshold=0.5, seg_frs=seg_frs)

    return stable_seg_time_range, seg_time_ranges, pvals, pval_times, pval_threshold


def gen_axes(fig):
    axes = {}
    axes['tfr'] = fig.add_subplot(411)
    axes['tfr'].set_ylabel("Trial firing rate (Hz)")

    axes['pval_tfr'] = fig.add_subplot(412, sharex=axes['tfr'])
    axes['pval_tfr'].set_ylabel("P-value")

    axes['amp'] = fig.add_subplot(413, sharex=axes['tfr'])
    axes['amp'].set_ylabel("Spike amplitude (mV)")

    axes['pval_amp'] = fig.add_subplot(414, sharex=axes['tfr'])
    axes['pval_amp'].set_ylabel("P-value")

    return axes


def plot_data(axes, trial_time_ranges, trial_firing_rates, seg_time_ranges, stable_seg_time_range, pvals, pval_times,
              spike_times, spike_amplitudes):
    def mark_segs(ax, segs, skip_unstable_segs=True):
        for seg_id, seg_range in segs.items():
            if skip_unstable_segs and seg_id < 0:
                continue
            if seg_range[0] == seg_range[1]:
                ax.axvline(seg_range[0], alpha=0.1)
            else:
                ax.axvspan(*seg_range, alpha=0.1)

    # plot trial firing rates
    axes['tfr'].plot(trial_time_ranges.mean(axis=1), trial_firing_rates, 'k+-')
    axes['tfr'].axhline(0.5, color='k', ls=':')
    axes['tfr'].set_ylim(ymin=0)

    # plot p-values of trial firing rate stability
    axes['pval_tfr'].plot(pval_times['tfr'], pvals['tfr'], 'k+-')
    axes['pval_tfr'].axhline(pval_threshold['tfr'], color='k', ls=':')
    axes['pval_tfr'].set_ylim(0, 1)

    # mark the segments obtained from trial firing rates
    mark_segs(axes['tfr'], seg_time_ranges['tfr'])
    mark_segs(axes['pval_tfr'], seg_time_ranges['tfr'])
    if stable_seg_time_range:
        axes['tfr'].axvspan(*stable_seg_time_range, ymin=0, ymax=0.5, color='red', alpha=0.1)
        axes['pval_tfr'].axvspan(*stable_seg_time_range, ymin=0, ymax=0.5, color='red', alpha=0.1)

    # plot spike amplitudes
    axes['amp'].plot(spike_times, np.abs(spike_amplitudes), 'k,')
    axes['amp'].axhline(spike_amplitude_threshold, color='k', ls=':')
    axes['amp'].set_ylim(ymin=0)

    # plot p-values of spike amplitude stability
    axes['pval_amp'].plot(pval_times['amp'], pvals['amp'], 'k+-')
    axes['pval_amp'].axhline(pval_threshold['amp'], color='k', ls=':')
    axes['pval_amp'].set_ylim(0, 1)

    # mark the segments obtained from spike amplitudes
    mark_segs(axes['amp'], seg_time_ranges['amp'])
    mark_segs(axes['pval_amp'], seg_time_ranges['amp'])


if __name__ == "__main__":
    # datadir = "/home/..."
    datadir = "/home/../"
    filenames = [
        # "segmentation_data_sample",
        "..."

    ]

    seg_params = {
        "TrialBinSize": 15,
        "TrialBinStep": 1,
        "SpikeBinSize": 100,
        "SpikeBinStep": 10,
        "FDR-q": 0.05,
    }

    for filename in filenames:
        fn = "{}/{}.hdf5".format(datadir, filename)
        with h5py.File(fn, 'r') as f:
            unit_ids = f['/units/list_unitIDs'][...]
            trial_time_ranges = f['/session/trial_times_ranges'][...]
            spike_amplitude_threshold = f['/session/spikethreshold'][...]
            for unit_id in unit_ids:
                # Load data
                group = "/units/{}/".format(unit_id)
                spike_times = f[group + 'spiketimes'][...]
                spike_amplitudes = f[group+'waveform_amplitudes'][...]

                print spike_times
                # print trial_time_ranges

                # Segment the spike train
                stable_seg_time_range, seg_time_ranges, pvals, pval_times, pval_threshold =\
                    find_stable_segment(spike_times, spike_amplitudes, trial_time_ranges, seg_params)

                """   
                Visualization part :
                """
                # Compute trial firing rates (used only for visualization)
                num_trial = len(trial_time_ranges)
                trial_firing_rates = np.empty(num_trial)
                for i_trial, (t_ini, t_fin) in enumerate(trial_time_ranges):
                    mask_trial = (t_ini <= spike_times) & (spike_times < t_fin)
                    trial_firing_rates[i_trial] = mask_trial.sum() / (t_fin-t_ini)

                # Generate plot
                fig = plt.figure(figsize=(10, 8))
                plt.suptitle("{}: {}\n"
                             "trial bin size: {}, trial bin step: {};\n"
                             "spike bin size: {}, spike bin step: {}".format(
                    filename, unit_id,
                    seg_params['TrialBinSize'], seg_params['TrialBinStep'],
                    seg_params['SpikeBinSize'], seg_params['SpikeBinStep'],
                ))
                axes = gen_axes(fig)
                plot_data(axes,
                          trial_time_ranges, trial_firing_rates,
                          seg_time_ranges, stable_seg_time_range,
                          pvals, pval_times,
                          spike_times, spike_amplitudes)

                plt.show()

