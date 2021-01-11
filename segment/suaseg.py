import os
import re
import datetime
import time
from collections import defaultdict

import numpy as np
import scipy
import scipy.stats
import scipy.spatial.distance
import sklearn.cluster

import odml


# functions for data loading
def load_class_header(filename_class, header_line=2):
    params = {}
    with open(filename_class, 'r') as f:
        for i in range(header_line-1):
            f.readline()
        for item in f.readline().split():
            name, value = item.split("=")
            # Some parameter names are followed by units in square brackets. Get rid of this.
            if name.endswith("]"):
                name = name.split("[")[0]
            try:
                params[name] = int(value)
            except ValueError:
                params[name] = float(value)
    return params


def find_filenames(datadir, subject, session, rec, filetype):
    if filetype not in ['imginfo', 'stimtiming', 'param', 'parameter', 'task', 'daq', 'lvd', 'odml', 'hdf5', 'RF']:
        raise ValueError("Filetype {0} is not supported.".format(filetype))

    if filetype in ['daq', 'lvd', 'hdf5', 'odml']:
        searchdir = "{dir}/{sbj}/{sess}".format(dir=datadir, sbj=subject, sess=session)
        re_filename = re.compile('{sess}.*_rec{rec}.*\.{filetype}$'.format(sess=session, rec=rec, filetype=filetype))
    elif filetype in ['RF',]:
        searchdir = "{dir}/{sbj}/{sess}".format(dir=datadir, sbj=subject, sess=session)
        re_filename = re.compile("{0}{1}.*".format(filetype, session))
    else:
        searchdir = "{dir}/{sbj}/{sess}/{sess}_rec{rec}".format(dir=datadir, sbj=subject, sess=session, rec=rec)
        re_filename = re.compile(".*{0}.*".format(filetype))

    filenames = os.listdir(searchdir)
    fn_found = []
    for fn in filenames:
        match = re_filename.match(fn)
        if match:
            fn_found.append("{0}/{1}".format(searchdir, fn))

    if len(fn_found) == 0:
        raise IOError("Files of type '{0}' not found.".format(filetype))
    else:
        return fn_found


def load_task(fn_task, blk=0):
    convfunc = lambda x: long(x)
    converters = {'INTERVAL': convfunc, 'TIMING_CLOCK': convfunc, 'GL_TIMER_VAL': convfunc}
    taskdata = np.genfromtxt(fn_task, skip_header=1, delimiter=',', names=True, dtype=None, converters=converters)
    if blk == 0:
        blockdata = taskdata
    else:
        blockdata = taskdata[taskdata['g_block_num'] == blk]

    evID = blockdata['log_task_ctrl']
    evtime = blockdata['TIMING_CLOCK']
    block = blockdata['g_block_num']
    trial = blockdata['TRIAL_NUM']

    num_trials = max(trial)
    success = []
    stimID = []
    task = []
    for i_trial in range(num_trials):
        trialID = i_trial + 1
        trialdata = blockdata[blockdata['TRIAL_NUM'] == trialID]
        success.append(trialdata[-1]['SF_FLG'])
        stimID.append(trialdata[0]['t_tgt_data'])
        task.append(trialdata[-1]['g_task_switch'])

    events = np.array(zip(evID, evtime, block, trial), dtype=[('evID', int), ('evtime', long), ('block', int), ('trial', int)])
    param = dict(num_trials=num_trials, success=success, stimID=stimID, task=task)
    return events, param


def identify_trial_time_ranges(task_events, task_param, sampling_rate):
    evID_img_on = task_param['task'][0] * 100 + 11
    evID_img_off = task_param['task'][0] * 100 + 12
    trial_time_ranges = []
    for i_trial in range(task_param["num_trials"]):
        # reject failure trials
        if task_param['success'][i_trial] <= 0:
            continue

        trialID = i_trial + 1
        trial_events = task_events[task_events['trial'] == trialID]

        # reject trials with missing image-onset or offset events
        if (evID_img_on not in trial_events['evID']) or (evID_img_off not in trial_events['evID']):
            continue

        t_ini = trial_events['evtime'][trial_events['evID'] == evID_img_on][0] / sampling_rate
        t_fin = trial_events['evtime'][trial_events['evID'] == evID_img_off][0] / sampling_rate
        trial_time_ranges.append([t_ini, t_fin])
    return np.array(trial_time_ranges)


# functions for spike train periodization+segmentation
def gap(data, refs=None, nrefs=10, ks=range(1, 11)):
    """
    # gap.py
    # (c) 2013 Mikael Vejdemo-Johansson
    # BSD License
    #
    # SciPy function to compute the gap statistic for evaluating k-means clustering.
    # Gap statistic defined in
    # Tibshirani, Walther, Hastie:
    #  Estimating the number of clusters in a data set via the gap statistic
    #  J. R. Statist. Soc. B (2001) 63, Part 2, pp 411-423

    Compute the Gap statistic for an nxm dataset in data.

    Either give a precomputed set of reference distributions in refs as an (n,m,k) scipy array,
    or state the number k of reference distributions in nrefs for automatic generation with a
    uniformed distribution within the bounding box of data.

    Give the list of k-values for which you want to compute the statistic in ks.
    """
    dst = scipy.spatial.distance.euclidean
    shape = data.shape
    if refs == None:
        tops = data.max(axis=0)
        bots = data.min(axis=0)
        dists = scipy.matrix(scipy.diag(tops-bots))

        rands = scipy.random.random_sample(size=(shape[0], shape[1], nrefs))
        for i in range(nrefs):
            rands[:, :, i] = rands[:, :, i]*dists + bots
    else:
        rands = refs

    gaps = scipy.zeros((len(ks),))
    for (i, k) in enumerate(ks):
        kmc, kml, _, = sklearn.cluster.k_means(data, k, n_init=1)
        disp = sum([dst(data[m, :], kmc[kml[m], :]) for m in range(shape[0])])

        refdisps = scipy.zeros((rands.shape[2],))
        for j in range(rands.shape[2]):
            kmc, kml, _, = sklearn.cluster.k_means(rands[:, :, j], k, n_init=1)
            refdisps[j] = sum([dst(rands[m, :, j], kmc[kml[m], :]) for m in range(shape[0])])
        gaps[i] = scipy.mean(scipy.log(refdisps)) - scipy.log(disp)
    return gaps


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


def smooth(data, window_size):
    data_padded = np.hstack([[data[0],] * ((window_size - 1) / 2), data, [data[-1], ] * ((window_size - 1) / 2)])
    window = np.bartlett(window_size)
    return np.convolve(data_padded, window/window.sum(), mode="valid")


def label_thresholded_segments(data, threshold):       # in segmentation : label_thresholded_segments(pvals, pval_threshold)
    condition = data >= threshold  # threshold = pval_threshold
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
        # assign positive segment IDs to stable rate segments
        segID = 1
        for i_seg in np.argsort([x[1]-x[0] for x in supth_seg_edges])[::-1]:
            seg_edges[segID] = supth_seg_edges[i_seg]
            segID += 1
    return seg_edges


def periodize_spike_train(spike_times, spike_sizes, params):
    bin_size = params["SpikeBinSize"]
    bin_step = params["SpikeBinStep"]
    num_spike = spike_times.size
    if num_spike < bin_size:
        raise ValueError("Spikes are fewer than the bin size.")

    # time-resolved estimation of cluster number
    bin_edges = np.arange(0, num_spike - bin_size, bin_step)
    num_bin = bin_edges.size
    unimodalities = np.zeros(num_bin)
    for i, idx_ini in enumerate(bin_edges):
        gaps = gap(spike_sizes[idx_ini:idx_ini+bin_size, np.newaxis], nrefs=params['GapNRefs'], ks=[1, 2])
        unimodalities[i] = gaps[0] / gaps[1]

    smoothing_size = bin_size/bin_step
    if smoothing_size % 2 == 0:
        smoothing_size -= 1
    unimodalities = smooth(unimodalities, smoothing_size)

    # detect period edges
    period_edges = label_thresholded_segments(unimodalities, 1.0)

    # convert the edge indices to times
    period_time_ranges = {}
    mean_unimodalities = {}
    for periodID, period_edge in period_edges.items():
        mean_unimodalities[periodID] = unimodalities[period_edge[0]:period_edge[1]].mean()
        idx_ini = 0 if period_edge[0] == 0 else bin_edges[period_edge[0]] + bin_size/2
        idx_fin = -1 if period_edge[1] == len(unimodalities) else bin_edges[period_edge[1]-1] + bin_size/2
        period_time_ranges[periodID] = [spike_times[idx_ini], spike_times[idx_fin]]
    mean_unimodalities['all'] = unimodalities.mean()

    # import matplotlib.pyplot as plt
    # plt.subplot(211)
    # plt.plot(bin_edges, unimodalities, 'm')
    # plt.axhline(1.0, color='gray')
    # for periodID, (idx_ini, idx_fin) in period_edges.items():
    #     print periodID, idx_ini, idx_fin
    #     t_ini = bin_edges[idx_ini]
    #     t_fin = bin_edges[idx_fin-1]
    #     spancolor = "red" if periodID >= 0 else "blue"
    #     plt.axvspan(t_ini, t_fin, color=spancolor, alpha=0.1)
    #     plt.plot((t_ini, t_fin), (mean_unimodalities[periodID], mean_unimodalities[periodID]), color=spancolor)
    # plt.subplot(212)
    # plt.plot(spike_times, spike_sizes, 'k,')
    # for periodID, (t_ini, t_fin) in period_time_ranges.items():
    #     print periodID, t_ini, t_fin
    #     spancolor = "red" if periodID >= 0 else "blue"
    #     plt.axvspan(t_ini, t_fin, color=spancolor, alpha=0.1)
    #     plt.plot((t_ini, t_fin), (mean_unimodalities[periodID], mean_unimodalities[periodID]), color=spancolor)
    # plt.show()

    return period_time_ranges, mean_unimodalities


def segment_spike_train(spike_times, trial_time_ranges, params):
    bin_size = params["TrialBinSize"]
    bin_step = params["TrialBinStep"]
    fdr_q = params["FDR-q"]
    num_trial = len(trial_time_ranges)
    trial_firing_rates = np.empty(num_trial)
    for i_trial, (t_ini, t_fin) in enumerate(trial_time_ranges):
        mask_trial = (t_ini <= spike_times) & (spike_times < t_fin)
        trial_firing_rates[i_trial] = mask_trial.sum() / (t_fin-t_ini)

    # TODO: Update in the collective toolbox file
    mask_ini = (spike_times[0] < trial_time_ranges[:, 0])
    mask_fin = (trial_time_ranges[:, 1] < spike_times[-1])
    if np.sum(mask_ini) == 0 or np.sum(mask_fin) == 0 :
        raise(ValueError, "spike-train is contained in too few trials.")

    idx_trial_ini = np.where(mask_ini)[0][0]
    idx_trial_fin = np.where(mask_fin)[0][-1] + 1
    if (idx_trial_fin - idx_trial_ini) < params["MinNumTrials"]:
        raise(ValueError, "spike-train is contained in too few trials.")

    bin_edges = np.arange(idx_trial_ini, idx_trial_fin - 2*bin_size + 1, bin_step)
    num_bin = bin_edges.size
    pvals = np.zeros(num_bin)
    pval_times = np.empty(num_bin)
    for i, idx_ini in enumerate(bin_edges):
        fr_pre = trial_firing_rates[idx_ini:idx_ini + bin_size]
        fr_post = trial_firing_rates[idx_ini + bin_size:idx_ini + 2*bin_size]
        _, pvals[i] = scipy.stats.ks_2samp(fr_pre, fr_post)
        pval_times[i] = (trial_time_ranges[idx_ini+bin_size-1][1] + trial_time_ranges[idx_ini+bin_size][0]) / 2

    # define the p-value threshold for FDR with Benjamini-Hochberg method
    pval_thresholds = np.linspace(fdr_q / len(pvals), fdr_q, len(pvals))
    idxs_rejected_pval = np.where(np.sort(pvals) < pval_thresholds)[0]
    pval_threshold = 0 if len(idxs_rejected_pval) == 0 else pval_thresholds[idxs_rejected_pval.max()]

    # detect segment edges
    seg_edges = label_thresholded_segments(pvals, pval_threshold)

    # convert the edge indices to times
    seg_time_ranges = {}
    mean_surprises = {}
    for segID, seg_edge in seg_edges.items():
        pvals_seg = pvals[seg_edge[0]:seg_edge[1]]
        mean_surprises[segID] = np.log10((1.0-pvals_seg) / pvals_seg).mean()
        t_ini = spike_times[0] if seg_edge[0] == 0 else pval_times[seg_edge[0]]
        t_fin = spike_times[-1] if seg_edge[1] == len(pvals) else pval_times[seg_edge[1]-1]
        seg_time_ranges[segID] = [t_ini, t_fin]

    # import matplotlib.pyplot as plt
    # plt.plot(spike_times[:-1], np.diff(spike_times), 'k,')
    # plt.plot(trial_time_ranges.mean(1), trial_firing_rates, 'k+-')
    # plt.plot(pval_times, np.log10((1.0-pvals)/pvals), 'm+-')
    # for segID, (t_ini, t_fin) in seg_time_ranges.items():
    #     spancolor = "blue" if segID >= 0 else "red"
    #     plt.axvspan(t_ini, t_fin, color=spancolor, alpha=0.1)
    # plt.show()

    return seg_time_ranges, mean_surprises


def suaseg(spike_times, spike_covs, spike_types, trial_time_ranges, params):
    unit_info = {}
    unitIDs = []
    for unitID in np.unique(spike_types):
        mask_unit = (spike_types == unitID)
        spike_times_unit = spike_times[mask_unit]
        spike_covs_unit = spike_covs[:, mask_unit]
        unit_ch = spike_covs_unit.mean(1).argmax()

        t_ini, t_fin = spike_times_unit[[0, -1]]
        num_trial = ((t_ini < trial_time_ranges[:, 0]) & (trial_time_ranges[:, 1] < t_fin)).sum()
        num_spike = spike_times_unit.size
        if num_trial < params["MinNumTrials"]:
            print "\tUnit {} is active in too few trials ({} spikes, {} trials).\n".format(unitID, num_spike, num_trial)
            continue
        if num_spike <= params["MinNumSpikes"]:
            print "\tUnit {} has too few spikes ({} spikes, {} trials).\n".format(unitID, num_spike, num_trial)
            continue

        print "\tProcessing unit {} ({} spikes, {} trials)...".format(unitID, num_spike, num_trial)

        # cut the spike train into periods according to spike size distribution
        period_time_ranges, mean_unimodalities = periodize_spike_train(spike_times_unit, spike_covs_unit[unit_ch], params)

        unitIDs.append(unitID)
        unit_info["Unit{}".format(unitID)] = {
            "NumSpikes": num_spike,
            "NumTrials": num_trial,
            "Start": t_ini,
            "End": t_fin,
            "MeanUnimodality": mean_unimodalities['all'],
        }

        periodIDs = []
        period_info = {}
        for periodID, (t_ini, t_fin) in period_time_ranges.items():
            num_trial = ((t_ini < trial_time_ranges[:, 0]) & (trial_time_ranges[:, 1] < t_fin)).sum()
            mask_period = mask_unit & (t_ini <= spike_times) & (spike_times <= t_fin)
            num_spike = mask_period.sum()
            if num_trial < params["MinNumTrials"]:
                print "\t\tPeriod {} contains too few trials ({} spikes, {} trials).".format(periodID, num_spike, num_trial)
                continue
            if num_spike < params["MinNumSpikes"]:
                print "\t\tPeriod {} contains too few spikes ({} spikes, {} trials).".format(periodID, num_spike, num_trial)
                continue

            print "\t\tProcessing period {} ({} spikes, {} trials)...".format(periodID, num_spike, num_trial)

            periodIDs.append(periodID)
            period_info["Period{}".format(periodID)] = {
                "NumSpikes": num_spike,
                "NumTrials": num_trial,
                "Start": t_ini,
                "End": t_fin,
                "MeanUnimodality": mean_unimodalities[periodID],
            }

            # cut the period into segments according to firing rate stability
            spike_times_period = spike_times[mask_period]
            seg_time_ranges, mean_surprises = segment_spike_train(spike_times_period, trial_time_ranges, params)

            segIDs = []
            seg_info = {}
            for segID, (t_ini, t_fin) in seg_time_ranges.items():
                num_trial = ((t_ini < trial_time_ranges[:, 0]) & (trial_time_ranges[:, 1] < t_fin)).sum()
                mask_seg = mask_period & (t_ini <= spike_times) & (spike_times <= t_fin)
                num_spike = mask_seg.sum()
                if num_trial < params["MinNumTrials"]:
                    print "\t\t\tSegment {} contains too few trials ({} spikes, {} trials).".format(segID, num_spike, num_trial)
                    continue
                if num_spike < params["MinNumSpikes"]:
                    print "\t\t\tSegment {} contains too few spikes ({} spikes, {} trials).".format(segID, num_spike, num_trial)
                    continue

                print "\t\t\tProcessing segment {} ({} spikes, {} trials)...".format(segID, num_spike, num_trial)
                segIDs.append(segID)
                seg_info["Segment{}".format(segID)] = {
                    "NumSpikes": num_spike,
                    "NumTrials": num_trial,
                    "Start": t_ini,
                    "End": t_fin,
                    "MeanSurprise": mean_surprises[segID],
                }
                print "\t\t\t...done."

            period_label = "Period{}".format(periodID)
            period_info[period_label]["SegmentIDs"] = segIDs
            period_info[period_label]["NumSegments"] = len(segIDs)
            period_info[period_label].update(seg_info)


            print "\t\t...done."

        unit_label = "Unit{}".format(unitID)
        unit_info[unit_label]["PeriodIDs"] = periodIDs
        unit_info[unit_label]["NumPeriods"] = len(periodIDs)
        unit_info[unit_label].update(period_info)

        print "\t...done.\n"

    unit_info["UnitIDs"] = unitIDs
    unit_info["NumUnits"] = len(unitIDs)

    return unit_info


# functions and a class for odML generation
def compute_properties(spike_times, spike_covs, params):
    rpv_threshold = params["RPVThreshold"]
    isis = np.diff(spike_times)
    rpv = (isis < rpv_threshold).sum() / np.float(isis.size)
    unit_ch = spike_covs.mean(1).argmax()
    snr = spike_covs[unit_ch,:].mean() / params["NoiseLevel"]
    return unit_ch, rpv, snr


def append_properties_to_unit_info(unit_info, spike_times, spike_covs, spike_types, params):
    rpv_threshold = params["RPVThreshold"]
    for unitID in unit_info["UnitIDs"]:
        mask_unit = (spike_types == unitID)
        channel, rpv, snr = compute_properties(spike_times[mask_unit], spike_covs[:, mask_unit], params)
        unit_label = "Unit{}".format(unitID)
        unit_info[unit_label].update({"Channel": channel, "RPV": rpv, "SNR": snr})

        for periodID in unit_info[unit_label]["PeriodIDs"]:
            period_label = "Period{}".format(periodID)
            t_ini = unit_info[unit_label][period_label]["Start"]
            t_fin = unit_info[unit_label][period_label]["End"]
            mask_period = mask_unit & (t_ini <= spike_times) & (spike_times <= t_fin)
            channel, rpv, snr = compute_properties(spike_times[mask_period], spike_covs[:, mask_period], params)
            unit_info[unit_label][period_label].update({"Channel": channel, "RPV": rpv, "SNR": snr})

            for segID in unit_info[unit_label][period_label]["SegmentIDs"]:
                seg_label = "Segment{}".format(segID)
                t_ini = unit_info[unit_label][period_label][seg_label]["Start"]
                t_fin = unit_info[unit_label][period_label][seg_label]["End"]
                mask_seg = mask_period & (t_ini <= spike_times) & (spike_times <= t_fin)
                channel, rpv, snr = compute_properties(spike_times[mask_seg], spike_covs[:, mask_seg], params)
                unit_info[unit_label][period_label][seg_label].update({"Channel": channel, "RPV": rpv, "SNR": snr})


def delete_keys_from_props(props, del_keys):
    for key in props:
        for prop in props[key]:
            for del_key in del_keys:
                del prop[del_key]


def convert_unit_info_to_odml_info(unit_info, params, units, dtypes):
    sectname = "Dataset/SpikeData"
    props = {sectname: []}
    for key in params.keys():
        props[sectname].append({"name": key, "value": params[key], "unit": units[key], "dtype": dtypes[key]})
    for key in ["NumUnits"]:
        props[sectname].append({"name": key, "value": unit_info[key], "unit": units[key], "dtype": dtypes[key]})
    if unit_info["NumUnits"] > 0:
        props[sectname].append({"name": "UnitIDs", "value": unit_info["UnitIDs"], "unit": None, "dtype": None})
    section_info = {
        "Dataset": {"name": "Dataset", "type": "dataset", "subsections": ["SpikeData",]},
        "Dataset/SpikeData": {"name": "SpikeData", "type": "dataset/neural_data", "subsections": []},
    }

    for unitID in unit_info["UnitIDs"]:
        unit_label = "Unit{}".format(unitID)
        sectname_unit = "{}/{}".format(sectname, unit_label)
        section_info[sectname]["subsections"].append(unit_label)
        section_info[sectname_unit] = {"name": unit_label, "type": "dataset/neural_data", "subsections": []}
        props[sectname_unit] = []
        for key in ["Channel", "NumSpikes", "NumTrials", "Start", "End", "MeanUnimodality", "RPV", "SNR", "NumPeriods"]:
            props[sectname_unit].append({"name": key, "value": unit_info[unit_label][key], "unit": units[key], "dtype": dtypes[key]})
        if unit_info[unit_label]["NumPeriods"] > 0:
            props[sectname_unit].append({"name": "PeriodIDs", "value": unit_info[unit_label]["PeriodIDs"], "unit": None, "dtype": None})

        for periodID in unit_info[unit_label]["PeriodIDs"]:
            period_label = "Period{}".format(periodID)
            sectname_period = "{}/{}".format(sectname_unit, period_label)
            section_info[sectname_unit]["subsections"].append(period_label)
            section_info[sectname_period] = {"name": period_label, "type": "dataset/neural_data", "subsections": []}
            props[sectname_period] = []
            for key in ["Channel", "NumSpikes", "NumTrials", "Start", "End", "MeanUnimodality", "RPV", "SNR", "NumSegments"]:
                props[sectname_period].append({"name": key, "value": unit_info[unit_label][period_label][key], "unit": units[key], "dtype": dtypes[key]})
            if unit_info[unit_label][period_label]["NumSegments"] > 0:
                props[sectname_period].append({"name": "SegmentIDs", "value": unit_info[unit_label][period_label]["SegmentIDs"], "unit": None, "dtype": None})

            for segID in unit_info[unit_label][period_label]["SegmentIDs"]:
                seg_label = "Segment{}".format(segID)
                sectname_seg = "{}/{}".format(sectname_period, seg_label)
                section_info[sectname_period]["subsections"].append(seg_label)
                section_info[sectname_seg] = {"name": seg_label, "type": "dataset/neural_data", "subsections": []}
                props[sectname_seg] = []
                for key in ["Channel", "NumSpikes", "NumTrials", "Start", "End", "MeanSurprise", "RPV", "SNR"]:
                    props[sectname_seg].append({"name": key, "value": unit_info[unit_label][period_label][seg_label][key], "unit": units[key], "dtype": dtypes[key]})

    return section_info, props


class odMLFactory(object):
    def __init__(self, section_info={}, default_props={}, filename='', strict=True):
        self.sect_info = section_info
        self.def_props = default_props
        self.strict = strict
        if filename:
            self._sections = self.__get_sections_from_file(filename)
        else:
            self._sections = {}
            for sectname in self.__get_top_section_names():
                self._sections[sectname] = self.__gen_section(sectname)

    def __get_sections_from_file(self, filename):
        # load odML from file
        with open(filename, 'r') as fd_odML:
            metadata = odml.tools.xmlparser.XMLReader().fromFile(fd_odML)
        sections = {}
        for sect in metadata.sections:
            sections[sect.name] = sect
        return sections

    def __get_top_section_names(self):
        topsectnames = []
        for key in self.sect_info:
            if '/' not in key:
                topsectnames.append(key)
        return topsectnames

    def __add_property(self, sect, prop, strict=True):
        if sect.contains(odml.Property(prop['name'], None)):
            sect.remove(sect.properties[prop['name']])
        elif strict is True:
            raise ValueError("Property '{0}' does not exist in section '{1}'.".format(prop['name'], sect.name))
        name = prop['name']
        if isinstance(prop['value'], list):
            value = prop['value']
        else:
            value = odml.Value(data=prop['value'], unit=prop['unit'], dtype=prop['dtype'])
        sect.append(odml.Property(name, value))

    def __gen_section(self, name, parent=''):
        longname = parent + name
        sect = odml.Section(name=name, type=self.sect_info[longname]['type'])

        # add properties
        if longname in self.def_props:
            for prop in self.def_props[longname]:
                self.__add_property(sect, prop, strict=False)

        # add subsections
        if 'subsections' in self.sect_info[longname]:
            for subsectname in self.sect_info[longname]['subsections']:
                sect.append(self.__gen_section(subsectname, longname+'/'))

        return sect

    def __get_section_from_longname(self, sectname):
        def get_subsect(sect, names):
            if len(names) == 0:
                return sect
            else:
                return get_subsect(sect.sections[names[0]], names[1:])

        names = sectname.split('/')
        if names[0] not in self._sections:
            return None
        else:
            return get_subsect(self._sections[names[0]], names[1:])

    def put_values(self, properties):
        for sectname, sectprops in properties.items():
            sect = self.__get_section_from_longname(sectname)
            if sect is None:
                raise ValueError("Invalid section name '{0}'".format(sectname))
            else:
                for prop in sectprops:
                    self.__add_property(sect, prop, strict=False)

    def get_odml(self, author, version=None):
        metadata = odml.Document(author, datetime.date.today(), version)
        for sect in self._sections.values():
            metadata.append(sect)
        return metadata

    def save_odml(self, filename, author, version=None):
        metadata = self.get_odml(author, version)
        odml.tools.xmlparser.XMLWriter(metadata).write_file(filename)


def print_metadata(metadata):
    def print_section(sect, ntab=0, tabstr='    '):
        tabs = tabstr * ntab
        print("{0}{1} (type: {2})".format(tabs, sect.name, sect.type))
        tabs = tabstr * (ntab + 1)
        for prop in sect.properties:
            if isinstance(prop.value, list):
                data = [str(x.data) for x in prop.value]
                if prop.value == []:
                    unit = ""
                    dtype = ""
                elif prop.value[0].unit is None:
                    unit = ""
                    dtype = prop.value[0].dtype
                else:
                    unit = prop.value[0].unit
                    dtype = prop.value[0].dtype
                print("{0}{1}: [{2}] {3} (dtype: {4})".format(tabs, prop.name, ', '.join(data), unit, dtype))
            else:
                unit = "" if prop.value.unit is None else prop.value.unit
                print("{0}{1}: {2} {3} (dtype: {4})".format(tabs, prop.name, prop.value.data, unit, prop.value.dtype))
        print

        for subsect in sect.sections:
            print_section(subsect, ntab+1)

    print("Version {0}, Created by {1} on {2}".format(metadata.version, metadata.author, metadata.date))
    print
    for sect in metadata.sections:
        print_section(sect)
        print


if __name__ == "__main__":
    # import parameters from the configuration file
    from suaseg_conf import *

    tic = time.time()

    for sbj, sess, rec, blk, site in datasets:
        dataset_name = "{}_rec{}_blk{}_{}_h".format(sess, rec, blk, site)
        print "\n{sbj}:{dataset_name} ".format(**locals())

        # load spike data and extract necessary information
        filename_class = "{dir}/{fn}.class_Cluster".format(dir=spikedir, sbj=sbj, fn=dataset_name)
        params["File"] = filename_class
        print "\tLoading spike data..."
        spike_data = np.genfromtxt(filename_class, skip_header=2, dtype=None, names=True)
        spike_params = load_class_header(filename_class)
        print "\t...done.\n"
        params["NoiseLevel"] = spike_params["NoiseLevel"] / 4.5
        spike_times = spike_data['event_time']
        spike_covs = np.array([spike_data["ch{}".format(i_ch)] for i_ch in range(params["NumChannels"])])
        spike_types = spike_data['type']

        # load task event data and extract necessary information
        filename_task = find_filenames(taskdir, sbj, sess, rec, 'task')[0]
        print "\tLoading task event data..."
        task_events, task_params = load_task(filename_task, blk)
        print "\t...done.\n"
        # time stamps in the task data file are relative to the beginning of the recording. Here the onset time of the
        # block is subtracted from the time stamps so that they are relative to the beginning of the block.
        task_events["evtime"] -= task_events["evtime"][0]
        trial_time_ranges = identify_trial_time_ranges(task_events, task_params, params["SamplingRate"])

        # periodize+segment spike trains and store the results in a dict: unit_info
        unit_info = suaseg(spike_times, spike_covs, spike_types, trial_time_ranges, params)

        # construct odML structure and save it in a file
        append_properties_to_unit_info(unit_info, spike_times, spike_covs, spike_types, params)
        section_info, props = convert_unit_info_to_odml_info(unit_info, params, odml_units, odml_dtypes)
        odml_factory = odMLFactory(section_info, strict=False)
        odml_factory.put_values(props)
        filename_odml = "{}/{}_SUA.odml".format(savedir, dataset_name)
        odml_factory.save_odml(filename_odml, odml_author, odml_version)
        print "\tSUA metadata saved in {0}\n".format(filename_odml)

        # # print out the odML structure for a check
        # print_metadata(odml_factory.get_odml(odml_author, odml_version))

        print "\tProcessing of {} done in {} secs.\n".format(dataset_name, time.time() - tic)
