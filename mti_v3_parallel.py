import numpy as np
import obspy
import os
import sys
sys.path.append('./moment_tensor_inversion/Zhao_and_Helmberger_1994/')
import compute_and_store_synths_w_seismograms_using_best_solution
sys.path.append('.')
from sven_utils import suFiles, suStationParser, suArray, suColors, suCorrelation, suTime, suLogic, suGeometry, suTrace, suMTI
from obspy.geodetics import gps2dist_azimuth
# import pylab as plt
from scipy.signal import ricker, triang, boxcar, unit_impulse
import pandas as pd
import itertools
from tqdm import tqdm
from multiprocessing import Process, Manager
from multiprocessing_generator import ParallelGenerator
import copy


import matplotlib
matplotlib.use('agg')
import pylab as plt

class Parameters(object):
    def __init__(self):
        ### Parameters: IO
        # Stream-file containing deconvolved, resampled, rotated traces, trimmed to event start-time.
        self.io_file_st = '/Users/sven/Documents/data/alland/alland_2017_11_08.mseed'
        self.io_file_st = '/Users/sven/Documents/data/alland/alland_preperoni_ALL.mseed'
        self.io_file_st = '/ssdraid/srvx8/sven/mti/alland_2017_11_08.mseed'
        self.io_file_st = '/ssdraid/srvx8/sven/mti/alland_preperoni_ALL.mseed'
        # List of stations. TODO: eventually replace functionality with Inventory()
        self.io_station_txt = '/Users/sven/Documents/data/imgw_stations/alland_stations_2017_11_08.txt'
        self.io_station_txt = '/Users/sven/Documents/data/imgw_stations/alland_stations.txt'
        self.io_station_txt = '/ssdraid/srvx8/sven/mti/alland_stations_2017_11_08.txt'
        self.io_station_txt = '/ssdraid/srvx8/sven/mti/alland_stations.txt'
        # Dir containting fundamental fault GFs
        self.io_gfdir = '/Users/sven/Documents/data/alland/mt_inv/gf_bases/'
        self.io_gfdir = '/ssdraid/srvx8/sven/mti/gf_bases/'
        # Result databases output directory
        self.io_outdir = '/Users/sven/Documents/data/alland/mt_inv/result_dbs/'
        self.io_outdir = '/ssdraid/srvx8/sven/mti/result_dbs/'
        # GF computating working dir and files
        self.io_wd = '/Users/sven/Documents/data/alland/mt_inv/wtf/.'
        self.io_wd = '/ssdraid/srvx8/sven/mti/gf_wd/.'
        # GF computation filenames for 1D model and temp distance file.
        self.mdl_f = 'slu.mod'
        self.dst_f = 'dist.tmp'
        # pre-determined unsuitable station/comps
        self.io_sta_comps_file = '/Users/sven/Documents/data/alland/mt_inv/stations_to_remove.npy'
        self.io_sta_comps_file = '/ssdraid/srvx8/sven/mti/stations_to_remove_2017_manual.npy'
        self.io_sta_comps_file = '/ssdraid/srvx8/sven/mti/stations_to_remove.npy'
        self.io_sta_comps_file = '/ssdraid/srvx8/sven/mti/result_dbs/mti_1105_v3_01_final_removal_7_(0.02, 0.05)_sta_comp_removed_iter4.npy'

        # waveform output directory
        self.io_waveform_out_dir = '/ssdraid/srvx8/sven/mti/waveforms/'
    
        ### Parameters: Parameter space to explore [min, max, spacing]
        # self.strike_params = [0, 181, 1]
        self.strike_params = [180, 361, 1]
        # self.strike_params = [0, 361, 1]
        self.dip_params = [0, 91, 1]
        # self.rake_params = [-90, 91, 1]
        # self.rake_params = [-180, 181, 1]
        self.rake_params = [-180, 181, 1]
        # Number of cores to use for parallelization
        self.n_cores = 150
        # Save detailed statistics or not
        # This contains the errors, magnitude, rms-misfit and timeshift determined for every single seismogram for every strike, dip, rake. 
        # WARNING: This can become VERY large and slows down the inversion/data handling. Depending on the size of your parameter space (strike, dip, rake, seismograms), this can take longer than the actual inversion itself.
        self.save_details = False
    
        ### Parameters: Source description
        # event description
        # self.source_coords = [48.11, 16.16]  # SLU LOC
        self.source_coords = [48.08, 16.07]  # ZAMG LOC
        self.source_time = obspy.UTCDateTime('2017-11-08T18:36:27')
        self.source_time = obspy.UTCDateTime('2016-04-25T10:28:22.9')
        
        ### Parameters: Green's Function parameters:
        # List of depths to invert for
        self.depths = np.arange(1, 16, 1)
        self.depths = [7]
        # self.depths = [0]
        # source time function stf: 't' triangle, 'i' dirac, ...
        self.stf = 'i'
        # source time function length in units of dt (e.g. at 5hz: 1*dt = 0.2s)
        self.stf_l = 1
        # Sampling rate of GFs (Stream needs to be the same)
        self.sampling_rate = 5

        ### Parameters: Waveform Fit
        # Which components to fit
        self.components = ['Z', 'R', 'T']
        # self.components = ['Z']
        # Filter range used to fit waveforms.
        # self.f_mins = [0.02, 0.03, 0.04, 0.05]
        # self.f_maxs = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        
        self.f_mins = [0.02] # , 0.03]
        self.f_maxs = [0.05] # , 0.1] 
        
        # self.f_mins = [0.3, 0.4, 0.5]
        # self.f_maxs = [0.6, 0.8, 1, 2]
        # self.f_mins = [0.03]
        # self.f_maxs = [0.1]
        # The methodology shifts Green's Functions in time to negate effect of local heterogeneities. SHould be lower than (.5 * 1/fmax) to avoid cycle skipping.
        self.max_shift_sec = 6
        # Distance scaling factor p (see Zhu & Helmberger)
        self.distance_scaling = {'Z': 1, 'R': 1, 'T': 1}
        self.distance_scaling = 1
        self.distance_scaling = .5
        # Reference Distance for distance scaling factor p
        self.distance_reference = 20E3  # [m]
        self.distance_reference = 100E3  # [m]
        # Definition of time windows to use. Choose wisely, tune using plot_XXX.py
        # Reference velocity of main phase of interest
        self.reference_velocity = 2.6
        # Time settings before theoretical arrival with reference velocity
        self.dist_scaler_pre = .2
        self.time_min_pre = 30
        self.time_max_pre = 75
        # Time settings after theoretical arrival with reference velocity
        self.dist_scaler_post = .2
        self.time_min_post = 30
        self.time_max_post = 75

        self.ptp_threshold = 3

        ### Parameters: Iterative selection of waveforms to keep
        # number of iterations to run
        self.n_iterations = 1
        # mean RMS-thresholds of waveforms to keep. Iteration #0 must be float('NAN')
        self.iteration_thresholds = [float('NAN'), 2, 2] # , 5, 3, 0.9]
        self.iteration_thresholds = [float('NAN'), 10, 2.5, 1.5, 1.0] # , 5, 3, 0.9]
        self.iteration_thresholds = [float('NAN'), 10, 3, 2, 1] # , 5, 3, 0.9]
        
        ### NAME YOUR PROJECT
        self.project_name = 'mti_0504_snr_estimate'
        self.project_name = 'mti_0504_snr_estimate_2'
        self.project_name = 'mti_0504_all_filters_SNR'
        self.project_name = 'mti_0503_filter_settings'
        self.project_name = 'mti_0711_02_still_wrong'
        self.project_name = 'mti_0711_03_iterations_test'
        self.project_name = 'mti_0711_04_new_iteration_run'
        self.project_name = 'mti_0712_compare_old_new_v2'
        self.project_name = 'mti_0712_az_baz_fix'
        self.project_name = 'mti_0712_03_rerun_no_iterations'
        self.project_name = 'mti_0713_01_old_time_windows_rough_grid'
        self.project_name = 'mti_0713_03_M0_on_unfilt_really'
        self.project_name = 'mti_0713_04_v2_seems_fixed'
        self.project_name = 'mti_0719_03_v2_with_correct_L1'
        self.project_name = 'mti_0720_v2_01_5_5_10_correct_err_mean'
        self.project_name = 'mti_0720_v2_02_5_5_10_correct_err_mean_local'
        self.project_name = 'mti_0720_v2_03_3_3_3_weekend_run'
        self.project_name = 'mti_0720_v3_01_parallel_new_db_style'
        self.project_name = 'mti_0720_v3_02_parallel_seems_working'
        self.project_name = 'mti_0721_v3_01_oof'
        self.project_name = 'mti_0723_v3_01_tqdm'
        self.project_name = 'mti_0723_v3_02_solution_space_seems_unstable'
        self.project_name = 'mti_0723_v3_03_2_2_2_30cores'
        self.project_name = 'mti_0723_v3_04_2_2_2_100cores_srvx2_test'
        self.project_name = 'mti_0723_v3_05_test_details_dict_rewrite'
        self.project_name = 'mti_0723_v3_07_2017_11_08_v1_same_removed_sta_comps'
        self.project_name = 'mti_0724_v3_01_details_dict_needs_fixing'
        self.project_name = 'mti_0724_v3_02_2017_testing_output_csvs'
        self.project_name = 'mti_0724_v3_03_main_2_2_2_speed_up_q_get'
        self.project_name = 'mti_0725_v3_01_main_2_2_2_full_details'
        self.project_name = 'mti_0725_v3_02_2017_run1'
        self.project_name = 'mti_0726_v3_01_2017_run2_did_not_save_details_correctly'
        self.project_name = 'mti_0726_v3_02_main_2_2_2_full_details_did_not_save_details_correctly'
        self.project_name = 'mti_0727_v3_01_main_2_2_2_switched_err_m0'
        self.project_name = 'mti_0727_v3_02_2017_2_2_2_switched_err_m0'
        self.project_name = 'mti_0727_v3_01_main_2_2_2_switched_err_m0_full_details'
        self.project_name = 'mti_0821_v3_01_main_depths'
        self.project_name = 'mti_1023_v3_01_depths_filterbands'
        self.project_name = 'mti_1024_v3_01_no_station_removal'
        self.project_name = 'mti_1024_v3_02_rms_removal'
        self.project_name = 'mti_1025_v3_01_rms_removal_with_first_ptp'
        self.project_name = 'mti_1025_v3_02_rms_removal_with_first_manual'
        self.project_name = 'mti_1025_v3_03_rms_removal_with_first_manual_save_stacomps'
        self.project_name = 'mti_1025_v3_04_rms_removal_weekend'
        self.project_name = 'mti_1029_v3_03_rms_removal_rerun_with_fixed_error'
        self.project_name = 'mti_1029_v3_04_save_stacomp_properly'
        self.project_name = 'mti_1029_v3_05_minimal_preselect'
        self.project_name = 'mti_1029_v3_06_plot_waveforms'
        self.project_name = 'mti_1029_v3_06_plot_waveforms3'
        self.project_name = 'mti_1029_v3_06_plot_waveforms4'
        self.project_name = 'mti_1029_v3_07_rms_based_on_proper_M0'
        self.project_name = 'mti_1029_v3_07_rms_based_on_proper_M0_2'
        self.project_name = 'mti_1029_v3_07_rms_actually_proper'
        self.project_name = 'mti_1030_v3_01_minimal_again'
        self.project_name = 'mti_1030_v3_02_estimate_final_error_on_all'
        self.project_name = 'mti_1030_v3_03_estimate_final_error_on_all'
        self.project_name = 'mti_1030_v3_04_estimate_final_error_on_all'
        self.project_name = 'mti_1030_v3_05_estimate_final_error_on_all'
        self.project_name = 'mti_1030_v3_06_minimal_again_again'
        self.project_name = 'mti_1030_v3_07_tuning_for_8'
        self.project_name = 'mti_1031_v3_01_tuning_for_8'
        self.project_name = 'mti_1031_v3_02_interlude_dist_scaling'
        self.project_name = 'mti_1031_v3_03_tuning_for_8'
        self.project_name = 'mti_1031_v3_04_tuning_for_8'
        self.project_name = 'mti_1031_v3_05_tuning_for_8'
        self.project_name = 'mti_1031_v3_06_rerun_with_initial_select'
        self.project_name = 'mti_1031_v3_07_tuning_for_8'
        self.project_name = 'mti_1031_v3_08_rerun_till_friday'
        self.project_name = 'mti_1102_v3_01_weekend_run_remove_bad'
        self.project_name = 'mti_1105_v3_01_final_removal'
        self.project_name = 'mti_1113_v3_01_minimal_final_test'
        self.project_name = 'mti_1122_v3_01_minimal_final_test_filterbands'
        self.project_name = 'mti_1219_v3_01_alternative_dc_space'
        self.project_name = 'mti_1219_v3_01_alternative_dc_space_only_final'
        self.project_name = 'mti_1219_v3_01_full_dc_space'
        self.project_name = 'mti_1219_v3_01_alternative_dc_space_only_final_from_1105'
        self.project_name = 'mti_1220_v3_01_full_solution_space_maybe'

        # Initializing some parameters. DO NOT CHANGE
        self.strikes = np.arange(
            self.strike_params[0],
            self.strike_params[1],
            self.strike_params[2]
            )

        self.dips = np.arange(
            self.dip_params[0],
            self.dip_params[1],
            self.dip_params[2]
            )

        self.rakes = np.arange(
            self.rake_params[0],
            self.rake_params[1],
            self.rake_params[2]
            )

        # columns to write in databases
        self.db_results_columns = ['STRIKE', 'DIP', 'RAKE', 'ERROR', 'M0']
        self.db_details_columns = ['STRIKE', 'DIP', 'RAKE', 'STATION', 'COMP', 'DT', 'eL1_1', 'eL1_2', 'eL2_1', 'eL2_2', 'e1_i', 'e2_i', 'M0']
        self.db_details_columns = ['STRIKE', 'DIP', 'RAKE', 'STATION', 'COMP', 'DT', 'ERROR', 'M0']

        # get list of stations from file
        self.stations = suStationParser.get_station_names_from_stationlist(stationfile=self.io_station_txt)

        # self.sta_comps_removed = [np.load(self.io_sta_comps_file), [], [], [], []]
        # self.sta_comps_removed = [np.load(self.io_sta_comps_file)]
        self.sta_comps_removed = [[] for _ in range(self.n_iterations)]

        self.keep_only = [
            'CONA.Z', 'CONA.R',
            'CSNA.Z', 'CSNA.R',
            'A009A.Z',
            'A013A.Z',
            'SOP.Z',
            'A001A.Z',
            'ARSA.Z', 'ARSA.R',
            'KRUC.Z', 'KRUC.R', 'KRUC.T',
            'A010A.Z',
            'A015A.Z',
            'A020A.Z',
            'A089A.Z',
            'VRAC.Z', 'VRAC.R', 'VRAC.T',
            'TREC.Z',
            'A021A.Z',
            'CKRC.Z',
            'A023A.Z',
            'A026A.Z',
            'A088A.Z',
            'SOKA.Z', 'SOKA.R',
            'A090A.Z',
            'A332A.Z',
            'GEC2.Z',
            'MORC.Z', 'MORC.R',
            'PRU.Z',
            'OKC.Z',
            'CRNS.Z']

        self.all_sta_comps = [f'{sta}.{comp}' for comp in ['Z', 'R', 'T'] for sta in self.stations]
        self.sta_comps_removed = [[sta_comp for sta_comp in self.all_sta_comps if sta_comp not in self.keep_only]]

        # self.sta_comps_removed = [[np.load('')]]
        self.sta_comps_removed = [np.load(self.io_sta_comps_file)]
        # print(len(self.sta_comps_removed), self.sta_comps_removed)
        # input('.')

        # compute distance, az, baz for all stations
        self.station_geometry = {station:suGeometry.get_distance_between_station_and_source(self.source_coords, station, self.io_station_txt) for station in self.stations}

        # compute distance scaling factors for all stations
        self.dist_scalings = {station: (self.station_geometry[station][0] / self.distance_reference) ** self.distance_scaling for station in self.stations}

        # compute the time windows for each station at which the seismograms and synthetics will be cut
        self.cut_time_windows = {
            station: suTime.get_time_window_to_cut(
                distance=self.station_geometry[station][0],
                parameters=self
                )
            for station in self.stations
            }

        # these will be used to write temporary information used throughout the program.
        self.current_filter_pair = (float('NAN'), float('NAN'))
        self.current_depth = float('NAN')
        self.current_iteration = float('NAN')
        self.io_current_f_main = None
        self.io_current_f_details = None
        self.current_sta_comps_in_stream = None


def parse_hpulse96(WD):
    """
    Parses an hpulse96 file and returns a dictionary containing all in _comps_ specified components.
    These components are the fundamental fault GFs. E.g. ['ZDD', 'RDD', 'ZDS', 'RDS', 'TDS', 'ZSS', 'RSS', 'TSS', ...].
    """
    # comps = ['ZDD', 'RDD', 'ZDS', 'RDS', 'TDS', 'ZSS', 'RSS', 'TSS']
    _comps = ['ZDD', 'RDD', 'ZDS', 'RDS', 'TDS', 'ZSS', 'RSS', 'TSS', 'ZEX', 'REX', 'ZVF', 'RVF', 'ZHF', 'RHF', 'THF']
    GF = {}
    with open('{0}/hpulse96.out'.format(WD)) as f:
        lines = f.read().splitlines()
        for line in lines[17:]:
            if line.split()[0] in _comps:
                comp = line.split()[0]
                GF[comp] = []
            values = line.split()
            if len(values) == 4 and values[-1] != '2048':
                for value in values:
                    GF[comp].append(float(value))
    
    return GF


def calc_A(station_az, strike, dip, rake):
    """
    Computes the radiation pattern coeffient A.
    Reference: Zhao & Helmberger 1994.
    """
    _Theta = np.deg2rad(station_az - strike)
    _delta = np.deg2rad(dip)
    _lambda = np.deg2rad(rake)

    A = []
    A.append(np.sin(2 * _Theta) * np.cos(_lambda) * np.sin(_delta) + .5 * np.cos(2 * _Theta) * np.sin(_lambda) * np.sin(2*_delta))
    A.append(np.cos(_Theta) * np.cos(_lambda) * np.cos(_delta) - np.sin(_Theta) * np.sin(_lambda) * np.cos(2 * _delta))
    A.append(.5 * np.sin(_lambda) * np.sin(2 * _delta))
    A.append(np.cos(2 * _Theta) * np.cos(_lambda) * np.sin(_delta) - .5 * np.sin(2 * _Theta) * np.sin(_lambda) * np.sin(2 * _delta))
    A.append(-1 * np.sin(_Theta) * np.cos(_lambda) * np.cos(_delta) - np.cos(_Theta) * np.sin(_lambda) * np.cos(2 * _delta))
    return A


def calc_synth(GF_BASES, A, comp):
    """
    Compute synthetic seismograms for given component.
    Needs pre-computed fundamental fault Green's Functions Dictionary.
    Reference: Zhao & Helmberger 1994
    """

    # Fundamental Fault weighting factors
    # A = calc_A(baz, strike, dip, rake)

    if comp == 'Z':
        ZSS = np.asarray(GF_BASES['ZSS'])
        ZDS = np.asarray(GF_BASES['ZDS'])
        ZDD = np.asarray(GF_BASES['ZDD'])
        g = ZSS * A[0] + ZDS * A[1] + ZDD * A[2]

    elif comp == 'R':
        RSS = np.asarray(GF_BASES['RSS'])
        RDS = np.asarray(GF_BASES['RDS'])
        RDD = np.asarray(GF_BASES['RDD'])
        g = RSS * A[0] + RDS * A[1] + RDD * A[2]

    elif comp == 'T':
        TSS = np.asarray(GF_BASES['TSS'])
        TDS = np.asarray(GF_BASES['TDS'])
        g = TSS * A[3] + TDS * A[4]

    else:
        raise ValueError

    # fmech96 uses seismic moment 1 dyne-cm to compute waveforms that are 1E20 smaller than what I have.
    # Therefore, I divide my synthetics by 1E20 to have the synthetics waveforms in [cm] as generated by a 1dyne-cm source.

    g /= 1E20  # convert to seismic moment of 1dyne-cm
    g /= 1E7  # convert seismic moment unit [dyne-cm] > [Nm]
    g /= 1E2  # convert displacement unit [cm] > [m]

    # We now have the synthetics as if created by a M0 = 1Nm source. We can use this to compute the Moment M0

    return g


# def calc_synth_as_tr(GF_BASES, distance, baz, strike, dip, rake, comp, parameters):
#     """
#     Return obspy.Trace() containing the synthetics, cut to correct length, depending on parameters.
#     """
#     synth = calc_synth(GF_BASES, baz, strike, dip, rake, comp)
# 
#     tr = obspy.Trace()
#     tr.stats.starttime = parameters.source_time
#     tr.stats.sampling_rate = parameters.sampling_rate
#     tr.data = synth
# 
#     return suTrace.cut_trace(tr, distance, parameters)



def shift_array_to_fit(x, y, fs, max_shift_sec=6):
    """
    Shifts the array y to match x within max_shift_sec seconds (based on cross_correlation coefficients).
    """
    # find best fit withing max_shift_sec seconds
    max_shift = max_shift_sec * fs + 1
    crosscorrelation = np.correlate(x, y, mode='same')
    crosscorrelation_causal = crosscorrelation[int(len(crosscorrelation) / 2):]
    crosscorrelation_acausal = crosscorrelation[:int(len(crosscorrelation) / 2)][::-1]
    if max(crosscorrelation_causal) > max(crosscorrelation_acausal):  # causal
        n_samples_shift = np.argmax(crosscorrelation_causal[:int(max_shift)])
    else:
        n_samples_shift = -np.argmax(crosscorrelation_acausal[:int(max_shift)])
    return suArray.shift_array(y, n_samples_shift), n_samples_shift / fs


def estimate_SNR(x, noise_window=.25):
    """
    Estimate signal-to-noise ratio of an array x, with SNR = |peak_amp| / std(noise).

    noise_window: ratio of x that is noise. E.g. .25 = last 25% of x.
    """
    peak_amplitude = np.max(np.abs(x))
    std_noise = np.std(x[-int(noise_window * len(x)):])
    snr = peak_amplitude / std_noise
    return snr


def sdr_grid_parallel(parameters):
    sdrs = []
    for strike in parameters.strikes:
        for dip in parameters.dips:
            for rake in parameters.rakes:
                sdrs.append([strike, dip, rake])
    return sdrs


def get_station_comp_in_stream(st, parameters):
    parameters.current_sta_comps_in_stream = [f'{tr.stats.station}.{tr.stats.channel[-1]}' for tr in st]


def generator_sta_comp(parameters, iteration=0):
    for station, comp in itertools.product(parameters.stations, parameters.components):
        if all('{0}.{1}'.format(station, comp) not in parameters.sta_comps_removed[i] for i in range(iteration + 1)) and '{0}.{1}'.format(station, comp) in parameters.current_sta_comps_in_stream:
            yield station, comp
        # else:    
        #     if '{0}.{1}'.format(station, comp) not in parameters.sta_comps_removed[iteration]:
        #         yield station, comp


def generator_sta_used(parameters, iteration=0):
    for station in parameters.stations:
        if suLogic.sta_used(station, parameters, iteration=iteration):
            yield station


def remove_unwanted_traces(st, parameters):
    # select traces to remove based on list in parameters.
    trs_to_remove = [tr for tr in st if any('{0}.{1}'.format(tr.stats.station, tr.stats.channel[-1]) in parameters.sta_comps_removed[i] for i in range(parameters.current_iteration + 1))]

    for _tr in trs_to_remove:
        st.remove(_tr)
    
    return st


def prepare_seismograms(st, parameters, do_filter=True):
    """ Returns an obspy.Stream() containing the prepared waveforms. """
    st_new = obspy.Stream()
    for tr in st:
        tr_new = prepare_tr(tr, parameters, do_filter)

        st_new += tr_new

    return st_new


def prepare_tr(tr, parameters, do_filter=True):
    """ Returns filtered and cut obspy.Trace() for one station. """
    # Apply bandpass filter
    if do_filter:
        tr.data = suArray.bandpass(
            tr.data,
            freqmin=parameters.current_filter_pair[0],
            freqmax=parameters.current_filter_pair[1],
            df=parameters.sampling_rate,
            corners=2,
            zerophase=False
            )

    # Cut Trace to correct length
    distance = parameters.station_geometry[tr.stats.station][0]
    tr = suTrace.cut_trace(tr, parameters)

    return tr


def load_GFs(parameters):
    return {station:suFiles.load_obj(get_gf_filename(station, parameters)) for station in parameters.stations if suLogic.sta_used(station, parameters, iteration=parameters.current_iteration)}


def calc_synth_as_tr(gfs_faults, A, station, comp, parameters):
    synth = calc_synth(GF_BASES=gfs_faults[station], A=A, comp=comp)
    tr = obspy.Trace()
    tr.data = synth
    tr.stats.station = station
    tr.stats.channel = 'S_{0}'.format(comp)
    tr.stats.sampling_rate = parameters.sampling_rate
    tr.stats.starttime = parameters.source_time
    return tr


def estimate_M0_for_all(seism_st, synth_st, parameters, iteration_override=False):
    M0s = {}
    M0_station_means = {}

    if iteration_override:
        active_generator = generator_sta_used(parameters, iteration=0)
    else:
        active_generator = generator_sta_used(parameters, iteration=parameters.current_iteration)

    for station in active_generator:

        _seism_st_tmp = seism_st.select(station=station)

        _M0_sta = []
        for _tr_seis in _seism_st_tmp:
            seism_data = _tr_seis.data

            comp = _tr_seis.stats.channel[-1]
            if comp in parameters.components:
                synth_data = synth_st.select(station=station, channel=f'S_{comp}')[0].data

                M0 = np.max(np.abs(seism_data)) / np.max(np.abs(synth_data))
                M0s[f'{station}.{comp}'] = M0
                _M0_sta.append(M0)

        M0_station_means[station] = np.mean(_M0_sta)
    
    return M0s, M0_station_means


def compute_synthetics_as_stream(GFs_fault_based, strike, dip, rake, parameters):
    synth_st = obspy.Stream()
    synth_st_M0 = obspy.Stream()
    for station, comp in generator_sta_comp(parameters, iteration=parameters.current_iteration):
        distance, az, baz = parameters.station_geometry[station]
            
        A = calc_A(baz, strike, dip, rake)
            
        synth_tr = calc_synth_as_tr(
            gfs_faults=GFs_fault_based,
            A=A,
            station=station,
            comp=comp,
            parameters=parameters
        )

        synth_tr_M0 = synth_tr.copy()

        synth_st += prepare_tr(synth_tr, parameters, do_filter=True)
        synth_st_M0 += prepare_tr(synth_tr_M0, parameters, do_filter=False)
    
    return synth_st, synth_st_M0


def perform_mti(strike, dip, rake, seism_st, seism_st_M0, GFs_fault_based, parameters, outdict_details):
    # 1. Compute synthetics
    synth_st, synth_st_M0 = compute_synthetics_as_stream(GFs_fault_based, strike, dip, rake, parameters)

    # 2. Estimate M0 per station and mean
    M0s, M0_station_means = estimate_M0_for_all(seism_st_M0, synth_st_M0, parameters)

    # 3. Estimate waveform match
    e_final, M0_total_mean, outdict_details = estimate_waveform_match(seism_st, synth_st, M0s, M0_station_means, strike, dip, rake, parameters, outdict_details)

    return e_final, M0_total_mean, outdict_details


def estimate_waveform_match(seism_st, synth_st, M0s, M0_station_means, strike, dip, rake, parameters, outdict_details):
    station_errors = []

    for station, comp in generator_sta_comp(parameters, iteration=parameters.current_iteration):
        # select data from streams. make copy to avoid altering original data
        seism_data = copy.deepcopy(seism_st.select(station=station).select(channel=f'HH{comp}')[0].data)
        synth_data = copy.deepcopy(synth_st.select(station=station).select(channel=f'S_{comp}')[0].data)

        # shift to fit
        synth_data, t_shift = shift_array_to_fit(
            x=seism_data,
            y=synth_data,
            fs=parameters.sampling_rate,
            max_shift_sec=parameters.max_shift_sec
            )

        # estimate error
        eL1_1, eL2_1, eL1_2, eL2_2, e1_i, e2_i = \
            suMTI.estimate_error_Zhao_Helmberger_1994(
                seismogram=seism_data,
                synthetic=synth_data,
                M0=M0s[f'{station}.{comp}'],
                M0_mean=M0_station_means[station],
                dist_scaling=parameters.dist_scalings[station]
                )
        
        
        # compute final error for this strike, dip, rake, station, component combination and save
        _error = np.mean([e1_i, e2_i])
        station_errors.append(_error)
        
        # plot waveforms for artefacts. WHAT'S GOING ON?!
#         if strike == 120 and dip == 55 and rake in [80]:
#             fig, ax = plt.subplots(1, 1)
#             ax.plot(seism_data, 'r')
#             ax.plot(synth_data * M0s[f'{station}.{comp}'], 'b')
#             
#             rms_misfit = suArray.rms_error_duputel_2012(copy.deepcopy(seism_data), copy.deepcopy(synth_data) * M0s[f'{station}.{comp}'])
#             ax.set_title(f'{rms_misfit:0.2f}')
#             fig.savefig(f'/raid/home/srvx7/sven/plots/mti/{station}_{comp}_{strike}_{dip}_{rake}.png')
#             plt.close(fig)

        if parameters.save_details:
            # compute rms misfit
            # rms_misfit = estimate_relative_RMS_misfit(seism_data, synth_data * M0s[f'{station}.{comp}'])
            # rms_misfit = suArray.rms_error_duputel_2012(copy.deepcopy(seism_data), copy.deepcopy(synth_data) * M0s[f'{station}.{comp}'])

            # np.save(f'/ssdraid/srvx8/sven/mti/{station}_{comp}.npy', [seism_data, synth_data])
            # input('')
            
            # add statistics to output details dictionary
            entries = [strike, dip, rake, station, comp, t_shift, _error, M0s[f'{station}.{comp}']]
            for col_id, col in enumerate(parameters.db_details_columns):
                outdict_details[col].append(entries[col_id])

    # communicate output dictionary for a given strike, dip, rake to the Queue.
    e_final = np.mean(station_errors)
    M0_total_mean = np.mean([M0s[_] for _ in M0s])

    return e_final, M0_total_mean, outdict_details


def estimate_waveform_match_for_all_for_best_solution(seism_st, synth_st, parameters):
    station_errors = []

    # compute M0 for best solution
    M0s, M0_station_means = estimate_M0_for_all(seism_st, synth_st, parameters, iteration_override=True)

    # for all initial channels, compute error estimate
    for station, comp in generator_sta_comp(parameters, iteration=0):
        # select data from streams. make copy to avoid altering original data
        seism_data = copy.deepcopy(seism_st.select(station=station).select(channel=f'HH{comp}')[0].data)
        synth_data = copy.deepcopy(synth_st.select(station=station).select(channel=f'S_{comp}')[0].data)

        # estimate error
        eL1_1, eL2_1, eL1_2, eL2_2, e1_i, e2_i = \
            suMTI.estimate_error_Zhao_Helmberger_1994(
                seismogram=seism_data,
                synthetic=synth_data,
                M0=M0s[f'{station}.{comp}'],
                M0_mean=M0_station_means[station],
                dist_scaling=parameters.dist_scalings[station]
                )
        
        
        # compute final error for this strike, dip, rake, station, component combination and save
        _error = np.mean([e1_i, e2_i])
        station_errors.append(_error)

    # communicate output dictionary for a given strike, dip, rake to the Queue.
    e_final = np.mean(station_errors)
    M0_total_mean = np.mean([M0s[_] for _ in M0s])

    return e_final, M0_total_mean


def parallel_mti(sdrs, seism_st, seism_st_M0, GFs_fault_based, parameters, q_main, q_details, _id):

    outdict_main = {col: [] for col in parameters.db_results_columns}
    if parameters.save_details:
        outdict_details = {col: [] for col in parameters.db_details_columns}
    else:
        outdict_details = None

    if _id == parameters.n_cores - 1:
        for sdr in tqdm(sdrs, mininterval=1):
            strike, dip, rake = sdr
            e_final, M0_total_mean, outdict_details = perform_mti(strike, dip, rake, seism_st, seism_st_M0, GFs_fault_based, parameters, outdict_details)

            entries = [strike, dip, rake, e_final, M0_total_mean]
            for col_id, col in enumerate(parameters.db_results_columns):
                outdict_main[col].append(entries[col_id])
    
    else:
        for sdr in sdrs:
            strike, dip, rake = sdr
            e_final, M0_total_mean, outdict_details = perform_mti(strike, dip, rake, seism_st, seism_st_M0, GFs_fault_based, parameters, outdict_details)
        
            entries = [strike, dip, rake, e_final, M0_total_mean]
            for col_id, col in enumerate(parameters.db_results_columns):
                outdict_main[col].append(entries[col_id])

    q_main.put(outdict_main)
    if parameters.save_details:
        q_details.put(outdict_details)
    
    # res_main = [strike, dip, rake, M0_total_mean, e_final]
    # q_main.put(res_main)

    # return res_main
    # parameters.io_current_f_main.writelines(f'{strike}\t{dip}\t{rake}\t{M0_total_mean}\t{e_final}')

    # print(return_dict['main'])
    # return results_db_dict, results_db_details_dict


def mti(parameters):
    print('\t:: Loading and Processing Seismograms')
    seism_st = remove_unwanted_traces(obspy.read(parameters.io_file_st), parameters)
    seism_st_M0 = prepare_seismograms(seism_st.copy(), parameters, do_filter=False)
    seism_st = prepare_seismograms(seism_st.copy(), parameters)

    get_station_comp_in_stream(seism_st, parameters)
    # print(parameters.current_sta_comps_in_stream)

    print("\t:: Loading base GFs")
    GFs_fault_based = load_GFs(parameters)
    # print(GFs_fault_based.keys())
    # input('.')

    print('\t:: Begin grid search')
    results_db_dict, results_db_details_dict = parallel_grid_search(
        seism_st,
        seism_st_M0,
        GFs_fault_based,
        parameters
    )
    
    print('\t:: Saving results')
    save_db(results_db_dict, parameters)
    if parameters.save_details:
        save_db(results_db_details_dict, parameters, add='_details')

def parallel_grid_search(seism_st, seism_st_M0, GFs_fault_based, parameters):
    """
    Divide grid search into jobs for each core, run the grid searches, and combine results from all of them.

    Input:
        * seism_st: obspy.Stream() containing all filtered and cut seismograms
        * seism_st_M0: obspy.Stream() containing all **unfiltered** and cut seismograms (needed for magnitude estimation)
        * GFs_fault_based: Dictionary

    
    Returns:
        * main_dict: Dictionary containing the final error and moment estimates for each strike, dip, rake
        * details_dict: Dictionary containing all information gathered during process: i.e. time shift, 

    """     
    manager = Manager()
    # Initialize two Queues, because two separate databses will be written.
    q_main = manager.Queue()
    if parameters.save_details:
        q_details = manager.Queue()
    else:
        q_details = None
    # Create strike, dip, rake, grid
    sdrs = sdr_grid_parallel(parameters)
    # Compute chunksize for each multiprocessing.Process
    sdr_chunksize = int(np.ceil(len(sdrs) / float(parameters.n_cores)))
    # With too many progress bars, they seems to bug out. TODO: explore why and fix.
    if parameters.n_cores > 1:
        print(f"\t:: Using {parameters.n_cores} cores. Displaying progress bar for Process #{parameters.n_cores}:")
    # Set up and start Process for each core. Each process handles a subset of the strike,dip,rake grid.
    procs = []
    for i in range(parameters.n_cores):
        p = Process(target=parallel_mti,
            args=(
                sdrs[sdr_chunksize * i:sdr_chunksize * (i + 1)],
                seism_st,
                seism_st_M0,
                GFs_fault_based,
                parameters,
                q_main,
                q_details,
                i,
                )
            )
        procs.append(p)
        p.start()

    print("\t:: Waiting for remaining Processes to finish and merge results:")

    # Merge main dictionaries of all Processes
    main_dict = initialize_db_dict(df_columns=parameters.db_results_columns)
    details_dict = initialize_db_dict(df_columns=parameters.db_details_columns)
    for i in tqdm(range(parameters.n_cores)):
        main_tmp_dct = q_main.get()
        main_dict = update_db_dict(main_dict, main_tmp_dct)
        if parameters.save_details:
            details_tmp_dct = q_details.get()
            details_dict = update_db_dict(details_dict, details_tmp_dct)
    # Merge details dictionaries of all Processes
    # for i in tqdm(range(q_details.qsize())):
    # Wait till all processes finished, then return dictionaries containing error estimates etc.
    for p in procs:
        p.join()
    return main_dict, details_dict


def update_db_dict(dct, new_dct):
    for col in new_dct:
        dct[col] = dct[col] + new_dct[col]
    return dct


def get_temp_db_fname(parameters, add):
    return '{0}/{1}{2}.csv'.format(parameters.io_outdir, get_identifier(parameters), add)


def save_db(db_dict, parameters, add=''):
    if add == '':
        cols = parameters.db_results_columns
    elif add == '_details':
        cols = parameters.db_details_columns
    db = pd.DataFrame.from_dict(db_dict)[cols].sort_values(by=['STRIKE', 'DIP', 'RAKE'])
    db.to_csv(get_temp_db_fname(parameters, add))

def initialize_db_dict(df_columns):
    return {col: [] for col in df_columns}


def get_identifier(parameters):
    return f'{parameters.project_name}_{parameters.current_depth}km_{parameters.current_filter_pair[0]}Hz_{parameters.current_filter_pair[1]}Hz_iter{parameters.current_iteration}'


def get_prev_identifier(parameters):
    return f'{parameters.project_name}_{parameters.current_depth}km_{parameters.current_filter_pair[0]}Hz_{parameters.current_filter_pair[1]}Hz_iter{parameters.current_iteration - 1}'


def get_gf_filename(station, parameters):
    return f'{parameters.io_gfdir}/GF_BASES_{station}_{parameters.current_depth}_{parameters.stf}_{parameters.mdl_f}.pkl'


def gf_exists(station, parameters):
    gf_file = get_gf_filename(station, parameters)
    return os.path.isfile(gf_file)


def compute_gfs(parameters):
    for station in parameters.stations:
        if not gf_exists(station, parameters):
            print('::Processing station {0}'.format(station))
            distance, az, baz = parameters.station_geometry[station]

            with open(f'{parameters.io_wd}/{parameters.dst_f}', 'w') as f:
                # f.writelines('{0} {1:0.1f} 2048 0 0\n'.format(round(distance/1000, 2), 1/parameters.sampling_rate))
                f.writelines('{0:0.2f} 0.2 2048 0 0'.format(distance/1000))

            # create gf preparations file
            CMD = 'hprep96 -M {0} -d {1} -HS {2:d} -HR 0'.format(parameters.mdl_f, parameters.dst_f, parameters.current_depth)
            os.system('cd {0}\n{1}'.format(parameters.io_wd, CMD))

            # compute gf in spectral domain
            CMD = 'hspec96 > hspec96.out'
            os.system('cd {0}\n{1}'.format(parameters.io_wd, CMD))

            # compute gf in time domain
            if parameters.stf == 'i':
                CMD = 'hpulse96 -D -{0} > hpulse96.out'.format(parameters.stf)
            else:
                CMD = 'hpulse96 -D -{0} -l {1} > hpulse96.out'.format(parameters.stf, parameters.stf_l)

            os.system('cd {0}\n{1}'.format(parameters.io_wd, CMD))

            gf_faults = parse_hpulse96(parameters.io_wd)

            suFiles.dump_obj(get_gf_filename(station, parameters), gf_faults)


def generator_filter_pair(parameters):
    for fmin, fmax in itertools.product(parameters.f_mins, parameters.f_maxs):
        if fmax >= 2*fmin:  # filter pair is somewhat broadband.
            yield (fmin, fmax)


def estimate_relative_RMS_misfit(x, y):
    """ Compute the relative RMS misift of two same-length arrays. """
    misfit = abs(x - y)
    return (np.linalg.norm(misfit, ord=2) / np.sqrt(np.linalg.norm(x, ord=2))) * (np.sqrt(np.linalg.norm(misfit, ord=2)) / np.linalg.norm(x, ord=2))


def find_best_solution_from_db(identifier, parameters):
    """Read database CSV, find and return best Strike, Dip, Rake solution for given current filter settings. """

    df = pd.read_csv('{0}/{1}.csv'.format(parameters.io_outdir, identifier))
    best = df.loc[df['ERROR'] == np.min(df['ERROR'])].iloc[0]
    best_solution = [best['STRIKE'], best['DIP'], best['RAKE'], best['M0']]

    return best_solution


def determine_sta_comps_removed(parameters):
    """
    Selection of seismograms, based on RMS misfit to best solution in previous iteration.
    Writes the seismogram-identifiers, which are not to be used, into parameters.sta_comps_removed
    """

    sta_comp_savefile = f'{parameters.io_outdir}/{parameters.project_name}_{parameters.current_depth}_{parameters.current_filter_pair}_sta_comp_removed_iter{parameters.current_iteration}.npy'

    if os.path.isfile(sta_comp_savefile):
        parameters.sta_comps_removed[parameters.current_iteration] = np.load(sta_comp_savefile)

        seism_st = remove_unwanted_traces(obspy.read(parameters.io_file_st), parameters)
        get_station_comp_in_stream(seism_st, parameters)

        return

    st_savefile = f'{parameters.io_waveform_out_dir}/{parameters.project_name}_{parameters.current_filter_pair[0]}Hz_{parameters.current_filter_pair[1]}Hz_{parameters.current_depth}km_iter{parameters.current_iteration - 1}_.MSEED'
    if os.path.isfile(st_savefile):
        stream = obspy.read(st_savefile)
        get_station_comp_in_stream(stream, parameters)

    else:
        raise FileNotFoundError('Waveform file from previous iteration not available')

    # seismograms, synthetics = compute_and_store_synths_w_seismograms_using_best_solution.run_from_mti(mti_parameters=parameters)
    # identifier = get_prev_identifier(parameters)
    # df_details = pd.read_csv('{0}/{1}_details.csv'.format(parameters.io_outdir, identifier))
    # solution = find_best_solution_from_db(identifier, parameters)
    
    parameters.sta_comps_removed[parameters.current_iteration] = []


    for station, comp in generator_sta_comp(parameters, iteration=parameters.current_iteration):
        # rms_misfit = df_details.loc[
        #     (df_details['STRIKE'] == int(solution[0])) &
        #     (df_details['DIP'] == int(solution[1])) &
        #     (df_details['RAKE'] == int(solution[2])) &
        #     (df_details['STATION'] == station) &
        #     (df_details['COMP'] == comp)
        #     ]['RMS_MISFIT'].values[0]

        rms_misfit = compute_rms_misfit_for_stacomp(station, comp, stream)

        if rms_misfit >= parameters.iteration_thresholds[parameters.current_iteration]:
            parameters.sta_comps_removed[parameters.current_iteration].append(f'{station}.{comp}')
    
    for station, comp in itertools.product(parameters.stations, parameters.components):
        if f'{station}.{comp}' in parameters.sta_comps_removed[parameters.current_iteration - 1] and not f'{station}.{comp}' in parameters.sta_comps_removed[parameters.current_iteration]:
            parameters.sta_comps_removed[parameters.current_iteration].append(f'{station}.{comp}')
    
    np.save(sta_comp_savefile, parameters.sta_comps_removed[parameters.current_iteration])


def compute_waveform_misfit_for_all_channels(seism_st, synth_st, parameters):
    """
    Computes and saves the misfit used during the inversion for all channels used in iteration 0.
    These channels include all, that pass the initial quality criteria.
    """
    for station, comp in generator_sta_comp(parameters, iteration=0):
        seism_data = copy.deepcopy(seism_st.select(station=station).select(channel=f'HH{comp}')[0].data)
        synth_data = copy.deepcopy(synth_st.select(station=station).select(channel=f'S_{comp}')[0].data)

        # shift to fit
        synth_data, t_shift = shift_array_to_fit(
            x=seism_data,
            y=synth_data,
            fs=parameters.sampling_rate,
            max_shift_sec=parameters.max_shift_sec
            )

        # estimate error
        eL1_1, eL2_1, eL1_2, eL2_2, e1_i, e2_i = \
            suMTI.estimate_error_Zhao_Helmberger_1994(
                seismogram=seism_data,
                synthetic=synth_data,
                M0=M0s[f'{station}.{comp}'],
                M0_mean=M0_station_means[station],
                dist_scaling=parameters.dist_scalings[station]
                )


def compute_rms_misfit_for_stacomp(station, comp, stream):
    if len(stream.select(station=station).select(channel='S_{0}'.format(comp))) > 0 and len(stream.select(station=station).select(channel='HH{0}'.format(comp))) > 0:
        tr_synth = stream.select(station=station).select(channel='S_{0}'.format(comp))[0]
        tr_seism = stream.select(station=station).select(channel='HH{0}'.format(comp))[0]
        rms_misfit = suArray.rms_error_duputel_2012(copy.deepcopy(tr_seism.data), copy.deepcopy(tr_synth.data))
    return rms_misfit

def output_waveforms_exists(parameters, label=''):
    """ Checks if the output .MSEED already exists. """
    st_savefile = f'{parameters.io_waveform_out_dir}/{parameters.project_name}_{parameters.current_filter_pair[0]}Hz_{parameters.current_filter_pair[1]}Hz_{parameters.current_depth}km_iter{parameters.current_iteration}_{label}.MSEED'
    return os.path.isfile(st_savefile)


def output_csv_exists(parameters):
    """ Checks if the output .csv already exists. """

    identifier = get_identifier(parameters)
    return os.path.isfile('{0}/{1}.csv'.format(parameters.io_outdir, identifier))


def determine_sta_comps_removed_first_pass(parameters):
    """
    Initial selection of seismograms, based on SNR.
    Writes the seismogram-identifiers, which are not to be used, into parameters.sta_comps_removed
    """
    # if inital selection specified otherwise, skip
    if len(parameters.sta_comps_removed[0]) >= 1:
        return

    seism_st = obspy.read(parameters.io_file_st)
    seism_st = prepare_seismograms(seism_st, parameters)
    
    parameters.sta_comps_removed[0] = []
    
    for tr in seism_st:
        # test #1 - peak to peak amplitude
        # if np.ptp(tr.data) / np.std(tr.data) < parameters.ptp_threshold:
        #     parameters.sta_comps_removed[0].append(f'{tr.stats.station}.{tr.stats.channel[-1]}')
        
        # remove AlpArray horizontals
        # if len(tr.stats.station) == 5 and tr.stats.station[0] == 'A' and tr.stats.channel[-1] in ['R', 'T']:
        #     parameters.sta_comps_removed[0].append(f'{tr.stats.station}.{tr.stats.channel[-1]}')

        # remove other band horizontals'
        # if tr.stats.station in ['CRNS', 'MOZS', 'OKC', 'ZALS', 'CSKK', 'MPLH', 'TREC', 'EGYH', 'SOP', 'GOPC', 'JAVC'] and tr.stats.channel[-1] in ['R', 'T']:
        #     parameters.sta_comps_removed[0].append(f'{tr.stats.station}.{tr.stats.channel[-1]}')

        # remove bad instrument response stations
        if tr.stats.station in ['MODS', 'ZST', 'VYHS', 'OBKA', 'EGYH', 'TIH', 'GROS', 'MOA']:
            parameters.sta_comps_removed[0].append(f'{tr.stats.station}.{tr.stats.channel[-1]}')

        # if tr.stats.station in ['MODS', 'ZST', 'VYHS', 'DOBS', 'A016A', 'A012A', 'A017A', 'A086A', 'A022A', 'TIH', 'MOA', 'A011A', 'A335A', 'A079A', 'GROS']:
        #     parameters.sta_comps_removed[0].append(f'{tr.stats.station}.{tr.stats.channel[-1]}')

        #remove bad traces
        # , 'DOBS', 'A005A', 'GROS', 'CSKK', 'A017A', 'A338A', 'A022A', 'MOA']


if __name__ == '__main__':
    # initialize parameters
    parameters = Parameters()

    for depth in parameters.depths:
        print('')
        print(f'  Depth {depth}km  '.center(os.get_terminal_size().columns, '*'))
        print('')
        print(f':: Computing GFs for {depth}km depth')
        parameters.current_depth = depth
        compute_gfs(parameters)

        for fp in generator_filter_pair(parameters):
            print('')
            print(f'  Filter Pair {fp}Hz  '.center(os.get_terminal_size().columns, '='))
            print('')
            print(f':: Starting Filter-pair {fp}')
            parameters.current_filter_pair = fp
            
            determine_sta_comps_removed_first_pass(parameters)
            print(f':: Initial Seismogram quality check removed ({len(parameters.sta_comps_removed[0])}): {parameters.sta_comps_removed[0]}')
            
            # save 
            np.save(f'{parameters.io_outdir}/{parameters.project_name}_{depth}_{fp}_sta_comp_removed_iter0.npy', parameters.sta_comps_removed[0])
            
            for iteration in range(parameters.n_iterations):
                print('')
                print(f'  Iteration #{iteration}  '.center(os.get_terminal_size().columns, '-'))
                print('')
                print(f':: RMS threshold {parameters.iteration_thresholds[iteration]}')
                parameters.current_iteration = iteration

                if iteration > 0:
                    # print(f":: Computing Seismograms from previous iteration's best solution")

                    print(f':: 0) Determining Seismograms to be eliminated from previous iteration')
                    determine_sta_comps_removed(parameters)
                    print(f':: 0) Removed Seismograms ({len(parameters.sta_comps_removed[parameters.current_iteration])}): {parameters.sta_comps_removed[parameters.current_iteration]}')

                if output_csv_exists(parameters):
                    print(f':: 1) Solution already available for {depth}km depth and {fp}Hz. Delete the .csvs if you want to rerun.')
                else:
                    print(f':: 1) Starting Grid Search for {depth}km depth and {fp}Hz')
                    mti(parameters)
                
                # if iteration == 0:
                #     _, _ = compute_and_store_synths_w_seismograms_using_best_solution.run_from_mti(mti_parameters=parameters)
                
                if output_waveforms_exists(parameters):
                    print(f':: 2) Waveforms already available for {depth}km depth and {fp}Hz. Delete the .MSEED if you want to rerun.')
                else:
                    print(f':: 2) Computing and Saving only used Synthetics/Seismograms for best solution.')
                    _, _ = compute_and_store_synths_w_seismograms_using_best_solution.run_from_mti(mti_parameters=parameters)

                if output_waveforms_exists(parameters, label='all') and os.path.isfile(f'{parameters.io_outdir}/{parameters.project_name}_{parameters.current_depth}_{parameters.current_filter_pair}_iter{parameters.current_iteration}_ERROR.npy'):
                    print(f':: 3) Waveforms and Error estimate already available for {depth}km depth and {fp}Hz. Delete the .MSEED and _ERROR.npy if you want to rerun.')
                else:
                    print(f':: 3) Computing and Saving ALL initial Synthetics/Seismograms for best solution and estimating Error.')
                    seismograms, synthetics = compute_and_store_synths_w_seismograms_using_best_solution.run_from_mti(mti_parameters=parameters, all_channels=True)
                
                    e_final, M0_total_mean = estimate_waveform_match_for_all_for_best_solution(seismograms, synthetics, parameters)

                    np.save(f'{parameters.io_outdir}/{parameters.project_name}_{parameters.current_depth}_{parameters.current_filter_pair}_iter{parameters.current_iteration}_ERROR.npy', e_final)