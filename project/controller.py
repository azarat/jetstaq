import project.model as db_model
from operator import itemgetter
from scipy import signal as signal_processor
from scipy.signal import find_peaks
import numpy as np
import sys
import matplotlib.pyplot as plt


class Controller:
    internal_shots = 0
    internal_time = 0
    internal_temperature_original = 0
    internal_channels_pos = 0
    internal_psi_r = 0
    internal_psi = 0
    internal_q_database = 0
    internal_q_x_database = 0
    internal_surfaces_ro = 0
    internal_surfaces_psinorm = 0

    win_list_names = [
        'triang',  # minimum info save
        'blackmanharris',  # middle info save (maybe best practice)
        'flattop',  # maximum info save
        'boxcar', 'blackman', 'hamming', 'hann',
        'bartlett', 'parzen', 'bohman', 'nuttall', 'barthann'
    ]

    database = db_model.LoadDB()

    def load(self):
        """ -----------------------------------------
             version: 0.10
             desc: load matlab DB (model trigger)
         ----------------------------------------- """
        sawdata = self.database.load()
        self.shots = self.database.shots

        return sawdata

    def assign(self, database, discharge, source):
        """ -----------------------------------------
             version: 0.10
             desc: assign data from loaded DB (model trigger)
         ----------------------------------------- """

        self.database.assign(database, discharge, source)

        self.channels_pos = self.database.channels
        self.time = self.database.time
        self.temperature_original = self.database.temperature
        self.psi = self.database.psi
        self.psi_r = self.database.psi_r
        self.q_database = self.database.q_database
        self.q_x_database = self.database.q_x_database
        self.surfaces_ro = self.database.surfaces_ro
        self.surfaces_psinorm = self.database.surfaces_psinorm

        return 1

    @property
    def q_database(self):
        return self.internal_q_database

    @q_database.setter
    def q_database(self, value):
        self.internal_q_database = value

    @property
    def q_x_database(self):
        return self.internal_q_x_database

    @q_x_database.setter
    def q_x_database(self, value):
        self.internal_q_x_database = value

    @property
    def surfaces_ro(self):
        return self.internal_surfaces_ro

    @surfaces_ro.setter
    def surfaces_ro(self, value):
        self.internal_surfaces_ro = value

    @property
    def surfaces_psinorm(self):
        return self.internal_surfaces_psinorm

    @surfaces_psinorm.setter
    def surfaces_psinorm(self, value):
        self.internal_surfaces_psinorm = value

    @property
    def shots(self):
        return self.internal_shots

    @shots.setter
    def shots(self, value):
        self.internal_shots = value

    @property
    def psi_r(self):
        return self.internal_psi_r

    @psi_r.setter
    def psi_r(self, value):
        self.internal_psi_r = value

    @property
    def psi(self):
        return self.internal_psi

    @psi.setter
    def psi(self, value):
        self.internal_psi = value

    @property
    def time(self):
        return self.internal_time

    @time.setter
    def time(self, value):
        self.internal_time = value

    @property
    def temperature_original(self):
        return self.internal_temperature_original

    @temperature_original.setter
    def temperature_original(self, value):
        self.internal_temperature_original = value

    @property
    def channels_pos(self):
        return self.internal_channels_pos

    @channels_pos.setter
    def channels_pos(self, value):
        self.internal_channels_pos = value


class Profiling(Controller):

    def order_by_r_maj(self, temperature_list_to_order, chan_pos_order_buffer):
        """ -----------------------------------------
            version: 0.2
            desc: ordering temperature list by r_maj position
            ;:param temperature_list_to_order: 2d array of temperature
            :return ordered 2d array
        ----------------------------------------- """
        temperature_ordered_list = []

        for channel in sorted(chan_pos_order_buffer.items(), key=itemgetter(1)):
            # if channel[0] in range(1, len(temperature_list_to_order)):
            if channel[0] in temperature_list_to_order:
                temperature_ordered_list.append(
                    temperature_list_to_order[channel[0]]
                )

        return temperature_ordered_list

    @staticmethod
    def normalization(temperature):

        """ -----------------------------------------
            version: 0.2
            desc: math normalization on 1
            :param temperature: 2d list of num
            :return normalized 2d list of num
        ----------------------------------------- """

        output = []

        for num_list in temperature:
            normalized = num_list / (sum(num_list[0:10]) / 10)
            output.append(normalized)

        return output

    @staticmethod
    def outlier_filter(temperature, boundary):

        """ -----------------------------------------
            version: 0.2.0
            desc: remove extra values from list
            :param temperature: 1d list of num
            :param boundary: array 0=>min and 1=>max
            :return filtered 1d list of num
        ----------------------------------------- """

        filtered_list = []

        for num_list in temperature:
            filtered_num = []
            for num in num_list:

                if num < boundary[0]:
                    filtered_num.append(boundary[0])
                elif num > boundary[1]:
                    filtered_num.append(boundary[1])
                else:
                    filtered_num.append(num)

            filtered_list.append(filtered_num)

        return filtered_list

    @staticmethod
    def outlier_filter_std_deviation(temperature_list, boundary, offset):

        """ -----------------------------------------
            version: 0.2.1
            desc: remove extra values from list
            :param temperature_list: 1d list of num
            :param boundary: float val intensity of cutting edge on standard deviation
            :param offset: int val of which temperature val reset
            :return filtered 1d list of num
        ----------------------------------------- """

        temperature_list = np.transpose(temperature_list)

        filtered_list = []

        for i_list, temperature in enumerate(temperature_list):
            filtered_num = []

            mean, data_std = np.mean(temperature), np.std(temperature)
            cut_off = data_std * boundary
            lower, upper = mean - cut_off, mean + cut_off

            for i, t in enumerate(temperature):
                if t < lower:
                    filtered_num.append(temperature[i-offset])
                    # filtered_num.append(lower)
                elif t > upper:
                    filtered_num.append(temperature[i-offset])
                    # filtered_num.append(upper)
                else:
                    filtered_num.append(t)

            filtered_list.append(filtered_num)

        return np.transpose(filtered_list)


class PreProfiling:

    @staticmethod
    def median_filtered(signal, threshold=3):
        """
        signal: is numpy array-like
        returns: signal, numpy array
        """
        difference = np.abs(signal - np.median(signal))
        median_difference = np.median(difference)
        s = 0 if median_difference == 0 else difference / float(median_difference)
        mask = s > threshold
        signal[mask] = np.median(signal)
        return signal

    @staticmethod
    def filter(input_signal, window_width, window_name):
        window = signal_processor.get_window(window_name, window_width)
        output_signal = []

        for temperature in input_signal:
            output_signal.append(
                signal_processor.convolve(temperature, window, mode='valid') / sum(window)
            )

        return output_signal

    @staticmethod
    def dict_to_list(dict_array):
        return [value for key, value in dict_array.items()]

    @staticmethod
    def list_to_dict(list_array):
        return {i: k for i, k in enumerate(list_array)}


class FindCollapseDuration:

    def collapse_duration(self, temperature_list_reverse, temperature_list, std_low_limit, inv_radius_channel, dynamic_outlier_limitation, median_filter_window_size):

        """ -----------------------------------------
            version: 0.3
            desc: search time points of Collapse duration
            :param temperature_list: 2d list of num
            :param time_list: 1d list of num
            :param std_low_limit: float val of min deviation which indicate start index
            :return list with int val of indexes in time_list
        ----------------------------------------- """

        collapse_start_time = self.collapse_start(temperature_list, median_filter_window_size, inv_radius_channel)
        collapse_end_time = self.collapse_end(temperature_list, inv_radius_channel, collapse_start_time, median_filter_window_size)

        # return (collapse_start_time, collapse_end_time)
        return (collapse_start_time, collapse_end_time)

    @staticmethod
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    # @staticmethod
    def collapse_end(self, temperature_list, r_inv_index, start, median_filter_window_size):
        """ -----------------------------------------
            version: 0.7
            desc: search time point at which end
            :param temperature_list: 2d list of num
            :return int val of index in time_list
        ----------------------------------------- """

        # fig, ax = plt.subplots()
        # fig.set_size_inches(15, 7)
        #
        # fig1, ax1 = plt.subplots()
        # fig1.set_size_inches(15, 7)

        temperature_list = temperature_list[r_inv_index:r_inv_index+10, median_filter_window_size[0]:-median_filter_window_size[0]]
        correlators = []
        height = []
        step_backward = 20
        smooth_window = 151
        smooth_window_offset = round((smooth_window - 1) / 2)
        for prototype_index, prototype in enumerate(temperature_list):

            window = signal_processor.get_window('triang', smooth_window)
            prototype = signal_processor.convolve(prototype, window, mode='valid')

            # ax.plot(prototype[(step_backward+smooth_window_offset):])

            ######################################### Financial momentum correlator
            correlator = []
            for t_i, t in enumerate(prototype):
                if t_i > step_backward:
                    # # Correlator based on all points in window
                    # k = []
                    # for i in range(step_backward):
                    #     k.append(t / prototype[t_i - i])
                    ##########################################

                    # # Correlator based only on endpoints
                    correlator.append(t / prototype[t_i - step_backward])

            correlator = correlator / np.mean(correlator[:500])
            correlator = np.abs(correlator - 1) + 1

            # ax1.plot(np.abs(correlator-1))

            correlator = correlator.tolist()
            correlators.append(correlator)
            ########################################## std correlator
            # correlator = []
            # for t_i, t in enumerate(prototype):
            #     if t_i > step_backward:
            #         correlator.append(np.std(prototype[t_i - step_backward:t_i]))
            #
            # correlator = correlator / np.mean(correlator[:500])
            # correlator = correlator.tolist()
            # correlators.append(correlator)
            ##########################################

            height.append(max(correlator) - 1)

        work_index = height.index(max(height))
        height = max(height)
        correlator = correlators[work_index]
        mu = correlator.index(max(correlator))
        level = (height / 4) + 1

        sigma_right = 0
        for c_i, c in enumerate(correlator[::-1]):
            if c > level:
                sigma_right = len(correlator) - c_i
                break

        sigma_left = 0
        for c_i, c in enumerate(correlator):
            if c > level:
                sigma_left = mu - c_i
                break

        # # Gaus, don't know if I need this
        # sigma = sigma_left + sigma_right
        # mu = np.round(((mu - sigma_left) + (mu + sigma_right)) / 2).astype(int)
        #
        # sigma = sigma / 4
        # x_values = np.linspace(0, len(correlator), len(correlator))
        # gaus = (height * self.gaussian(x_values, mu, sigma)) + 1
        #
        # deviation = 0.0145 * height
        # end = 0
        # for g_i, g in enumerate(gaus[::-1]):
        #     if (g - 1) > deviation:
        #         end = len(correlator) - g_i + step_backward + smooth_window_offset  # !!!!!!!!!!!!!!!!11
        #         break
        ##########################################


        ########################################## DEBUG ploting
        # fig, ax = plt.subplots()
        # fig.set_size_inches(15, 7)
        # prototype = temperature_list[work_index]
        # prototype = prototype / np.mean(prototype[:500])
        # ax.plot(correlator)
        # ax.plot(prototype[(step_backward+smooth_window_offset):])
        ##########################################

        end = sigma_right + step_backward + smooth_window_offset
        return end

    # @staticmethod
    def collapse_start(self, temperature_list, median_filter_window_size, r_inv_index=60):
        """ -----------------------------------------
            version: 0.7
            desc: search time point at which start Precursor Phase
            :param temperature_list: 2d list of num
            :return int val of index in time_list
        ----------------------------------------- """

        fig, ax = plt.subplots(2, 2)
        fig.set_size_inches(15, 7)

        temperature_list = temperature_list[:r_inv_index-3, median_filter_window_size[0]:-median_filter_window_size[0]]
        correlators = []
        height = []
        prominence = []
        step_backward = 21
        smooth_window = 51
        medfit_width = 51
        smooth_window_offset = round(smooth_window / 2)
        step_backward_offset = round(step_backward / 2)
        for prototype_index, prototype in enumerate(temperature_list):

            prototype = signal_processor.medfilt(prototype, medfit_width)
            window = signal_processor.get_window('triang', smooth_window)
            prototype = signal_processor.convolve(prototype, window, mode='valid')

            ax[0, 0].plot(prototype[(step_backward+smooth_window_offset):])

            """ Financial momentum correlator """
            correlator = []
            for t_i, t in enumerate(prototype):
                if t_i > step_backward:
                    """ Correlator based on all points in window """
                    # k = []
                    # for i in range(step_backward):
                    #     k.append(t / prototype[t_i - i])
                    # correlator.append(np.mean(k))
                    ##########################################

                    """ Correlator based only on endpoints """
                    correlator.append(prototype[t_i - step_backward] / t)

            correlator = correlator / np.mean(correlator[:500])
            correlator = np.abs(correlator - 1) + 1

            ax[0, 1].plot(correlator)

            correlator = correlator.tolist()
            correlators.append(correlator)

            """ Std correlator """
            # correlator = []
            # for t_i, t in enumerate(prototype):
            #     if t_i > step_backward:
            #         correlator.append(np.std(prototype[t_i - step_backward:t_i]))
            #
            # correlator = correlator / np.mean(correlator[:500])
            # correlator = correlator.tolist()
            # correlators.append(correlator)
            ##########################################

            """ Determine height of current correlator """
            height_val = max(correlator) - 1
            height.append(height_val)

            """ Determine prominence of current correlator """
            peaks, _ = find_peaks(correlator, prominence=(height_val / 1.5))
            correlator = np.asarray(correlator)
            prominence.append(np.mean(correlator[peaks]))

        """ Determine work_index by max value of max prominence in whole set of correlators
            Which one is better??? """
        work_index = prominence.index(max(prominence))
        """ Determine work_index by max height in whole set of correlators """
        # work_index = height.index(max(height))

        height = max(height)
        correlator = correlators[work_index]
        mu = correlator.index(max(correlator))

        level_height = (height / 5) + 1
        peaks, _ = find_peaks(correlator[:500])
        mean_peak_height = np.mean(np.asarray(correlator)[peaks])

        level_std = ((mean_peak_height - 1) * 10) + np.mean(correlator[:500])
        level = level_height if level_std > level_height else level_std

        sigma_right = 0
        for c_i, c in enumerate(correlator[::-1]):
            if c > level:
                sigma_right = len(correlator) - c_i
                break

        sigma_left = 0
        for c_i, c in enumerate(correlator):
            if c > level:
                sigma_left = c_i
                break

        """ Gaus, don't know if I need this """
        # sigma = sigma_left + sigma_right
        # mu = np.round(((mu - sigma_left) + (mu + sigma_right)) / 2).astype(int)
        #
        # sigma = sigma / 4
        # x_values = np.linspace(0, len(correlator), len(correlator))
        # gaus = (height * self.gaussian(x_values, mu, sigma)) + 1
        #
        # deviation = 0.0145 * height
        # end = 0
        # for g_i, g in enumerate(gaus[::-1]):
        #     if (g - 1) > deviation:
        #         end = len(correlator) - g_i + step_backward + smooth_window_offset  # !!!!!!!!!!!!!!!!11
        #         break
        ##########################################

        """ DEBUG ploting etc. """
        # # print(sigma_left)
        # prototype = temperature_list[work_index]
        # prototype = prototype / np.mean(prototype[:500])
        #
        # window = signal_processor.get_window('triang', smooth_window)
        # prototype_smooth = signal_processor.convolve(temperature_list[work_index], window, mode='valid')
        # prototype_smooth = prototype_smooth / np.mean(prototype_smooth[:500])
        #
        # ax[1, 0].plot(prototype_smooth)
        # ax[1, 1].plot(prototype[(step_backward+smooth_window_offset):])
        # ax[1, 1].plot([level for x in correlator])
        # ax[1, 1].plot(correlator)
        #
        # peaks, _ = find_peaks(correlator, prominence=(height / 1.5))
        # correlator = np.asarray(correlator)
        # plt.plot(correlator)
        # plt.plot(peaks, correlator[peaks], "x")
        ##########################################

        start = sigma_left + step_backward + smooth_window_offset - step_backward_offset
        return start


class FindInvRadius:

    @staticmethod
    def trend_indicator(plane_list):

        """ -----------------------------------------
            version: 0.2
            desc: define if list of nums (T(r_maj)) increase or decrease
            :param plane_list: 1d list of num
            :return indicator => 1: increase, -1: decrease, 0: flat/undefined
        ----------------------------------------- """

        compare_num = plane_list[0]
        increase = []
        decrease = []
        stat_weight = 1 / len(plane_list)

        for num in plane_list:

            difference = num - compare_num
            compare_num = num

            if difference > 0:
                increase.append(difference)
            elif difference < 0:
                decrease.append(difference)

        indicator_increase = sum(map(abs, increase)) * (len(increase) * stat_weight) + 1
        indicator_decrease = sum(map(abs, decrease)) * (len(decrease) * stat_weight) + 1

        if indicator_increase > indicator_decrease:
            indicator = 1
        elif indicator_decrease > indicator_increase:
            indicator = -1
        else:
            indicator = 0

        return indicator

    def inv_radius(self, temperature_list, window_width, std_low_limit, channel_offset):

        """ -----------------------------------------
            version: 0.2
            desc: define if list of nums increase or decrease
            :param temperature_list: 2d array of nums normalised on 1
            :param std_low_limit: float val to skip flat regions
            :param window_width: int val of len by which make plane indicating
            :return main_candidate_index: value of the most probable index
                    of channel with inversion radius
        ----------------------------------------- """

        temperature_list = np.transpose(temperature_list)
        mean = np.mean(temperature_list[0])
        area = int((len(temperature_list[0]) - (len(temperature_list[0]) % window_width)))
        candidate_list = []
        stat_weight = {}

        for timeline, t_list in enumerate(temperature_list):
            if timeline == 0:
                continue

            # flat_outlier = sum(abs(t_list - mean)) / len(t_list)
            flat_outlier = np.std(t_list)

            # print(flat_outlier, ' ', std_low_limit)
            if flat_outlier > std_low_limit:
                candidates = []
                plane_area_direction_prev = 1
                for i in range(window_width, area, window_width):
                    if i < channel_offset:
                        continue

                    analysis_area = t_list[i-1:i + window_width]

                    """ Analysis only upward trends """
                    plane_area_direction = self.trend_indicator(analysis_area)
                    if plane_area_direction != -1 and plane_area_direction_prev == 1:

                        """ Analysis only analysis_area which have intersection with mean value"""
                        upper_area = 0
                        under_area = 0
                        for t_i, t_analysis in enumerate(analysis_area):
                            upper_area = 1 if t_analysis > mean else upper_area
                            under_area = 1 if t_analysis < mean else under_area

                        """ Candidates => (range of points, temperature at each point) """
                        if upper_area == 1 and under_area == 1:
                            candidates.append((range(i, i + window_width), analysis_area))
                            stat_weight_to_update = (stat_weight[i] + 1) if i in stat_weight else 0
                            stat_weight.update({i: stat_weight_to_update})

                    plane_area_direction_prev = plane_area_direction

                    """ Candidate_list => (timeline with candidates, candidates) """
                    candidate_list.append((timeline, candidates))

        if len(stat_weight) == 0:
            # print('Error: std_low_limit is too big')
            return 0

        search_area_max = max(stat_weight.items(), key=itemgetter(1))
        search_area = search_area_max[0]

        # candidate_info = 0
        # if search_area_max[1] < 1000:
        #     return candidate_info

        temperature_list = np.transpose(temperature_list)
        main_candidate_index = self.sum_deviation(temperature_list[search_area:(search_area + window_width)])
        main_candidate_index = search_area + main_candidate_index

        return main_candidate_index

    def inv_radius_intersection(self, temperature_list, window_width, std_low_limit, r_maj):

        """ -----------------------------------------
            version: 0.2.1
            desc: define if list of nums increase or decrease
            :param temperature_list: 2d array of nums normalised on 1
            :param std_low_limit: float val to skip flat regions
            :param window_width: int val of len by which make plane indicating
            :return main_candidate_index: value of the most probable index
                    of channel with inversion radius
        ----------------------------------------- """

        temperature_list = np.transpose(temperature_list)
        mean = sum(temperature_list[0]) / len(temperature_list[0])  # normalised on 1
        area = int((len(temperature_list[0]) - (len(temperature_list[0]) % window_width)))
        stat_weight = {}
        candidates = []

        for timeline, t_list in enumerate(temperature_list):
            flat_outlier = sum(abs(t_list - mean)) / len(t_list)

            if flat_outlier > std_low_limit:
                plane_area_direction_prev = 1
                for i in range(window_width, area, window_width):

                    analysis_area = t_list[i-1:i + window_width]

                    """ Analysis only upward trends """
                    plane_area_direction = self.trend_indicator(analysis_area)
                    if plane_area_direction != -1 and plane_area_direction_prev == 1:

                        """ Analysis only analysis_area which have intersection with mean value"""
                        upper_area = 0
                        under_area = 0
                        intersection_indexes = ()
                        for tia, t_analysis in enumerate(analysis_area):
                            upper_area = 1 if t_analysis > mean else upper_area
                            if under_area == 1 and upper_area == 1:
                                intersection_indexes = (i + tia - 2, i + tia - 1)
                                break
                            under_area = 1 if t_analysis < mean else under_area

                        """ Candidates => (range of points, temperature at each of point) """
                        if upper_area == 1 and under_area == 1 and len(intersection_indexes) == 2:

                            intersection = self.intersection_pos(t_list, intersection_indexes, r_maj, mean)
                            if intersection > 0:
                                candidates.append((intersection_indexes[0], intersection))

                                stat_weight_to_update = (stat_weight[intersection_indexes[0]] + 1) if intersection_indexes[0] in stat_weight else 1
                                stat_weight.update({intersection_indexes[0]: stat_weight_to_update})

                    plane_area_direction_prev = plane_area_direction

        sorted_candidates = {}
        # print(len(candidates))
        for c in candidates:
            inter_list = sorted_candidates[c[0]] if c[0] in sorted_candidates else []
            inter_list.append(c[1])
            sorted_candidates.update({c[0]: inter_list})

        if len(stat_weight) == 0:
            sys.exit('Error: std_low_limit is too big')

        search_area_max = max(stat_weight.items(), key=itemgetter(1))
        search_area = search_area_max[0]

        intersection = np.mean(sorted_candidates[search_area])

        candidate_info = (0, 0)
        if search_area_max[1] < 500:
            return candidate_info

        candidate_info = ('{:.4f}'.format(intersection), str(search_area) + "-" + str(search_area + 1))

        return candidate_info

    @staticmethod
    def intersection_pos(t_list, intersection_indexes, r_maj, mean):

        """ -----------------------------------------
            version: 0.2
            desc: calculate deviation from mean val and give index of channel
            :param t_list:
            :param intersection_indexes:
            :param r_maj:
            :param mean:
            :return intersection float val
        ----------------------------------------- """

        cathete_under = mean - t_list[intersection_indexes[0]]
        cathete_above = t_list[intersection_indexes[1]] - mean

        cathete_side = cathete_under + cathete_above

        if cathete_side > 0:
            cathete_down = r_maj[intersection_indexes[1]] - r_maj[intersection_indexes[0]]

            hypotenuse = np.sqrt(np.power(cathete_side, 2) + np.power(cathete_down, 2))

            aspect_ratio = hypotenuse / cathete_side
            hypotenuse_under = aspect_ratio * cathete_under

            cathete_mean = np.sqrt(np.power(hypotenuse_under, 2) - np.power(cathete_under, 2))

            intersection = r_maj[intersection_indexes[0]] + cathete_mean
        else:
            intersection = 0

        return intersection

    @staticmethod
    def sum_deviation(search_area):

        """ -----------------------------------------
            version: 0.3
            desc: calculate deviation from mean val and give index of channel
            :param search_area: 2d array of nums normalised on 1 where can be channel with inv radius
            :return index of channel with minimum deviation from mean that means
                    that it is index of inversion radius
        ----------------------------------------- """

        deviation = []
        for t_list in search_area:
            deviation.append(np.std(t_list))

        return deviation.index(min(deviation))


class MachineLearning:

    @staticmethod
    def ml_load(filename):
        """ -----------------------------------------
             version: 0.3
             desc: load data from matlab with previously prepared data
             :param filename: string val
             :return nd array
         ----------------------------------------- """
        db = db_model.Model()
        mat = db.load(filename)

        data = {
            'ece_data': mat['ece_data'][0, 0]['signal'],
            'discharge': mat['ece_data'][0, 0]['discharge']
        }

        return data

    @staticmethod
    def ml_find_inv_radius(data_train, data_test):
        from sklearn.neural_network import MLPClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

        data_test = np.array(data_test)[1000:1900, :60]

        print('data_train len:', len(data_train))

        X_train = data_train[:, :-1]
        y_train = data_train[:, -1]

        # KNeighborsClassifier(3),
        # SVC(kernel="linear", C=0.025),
        # SVC(gamma=2, C=1),
        # GaussianProcessClassifier(1.0 * RBF(1.0)),
        # DecisionTreeClassifier(max_depth=5),
        # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        # MLPClassifier(alpha=1),
        # AdaBoostClassifier(),
        # GaussianNB(),
        # QuadraticDiscriminantAnalysis()
        model = DecisionTreeClassifier().fit(X_train, y_train)

        print('Accuracy of Linear SVC classifier on training set: {:.2f}'
         .format(model.score(X_train, y_train)))

        return model.predict(data_test)

    @staticmethod
    def ml_find_collapse_duration(data_train, data_test):
        from sklearn.tree import DecisionTreeClassifier

        data_test = np.array(data_test)[15:45, 1000:1901]

        print('data_train len:', len(data_train))
        # print(data_train.shape)
        # exit()

        X_train = data_train[:, :-1]
        y_train = data_train[:, -1]
        # print(y_train[0])
        # exit()

        model = DecisionTreeClassifier().fit(X_train, y_train)

        print('Accuracy of Linear SVC classifier on training set: {:.2f}'
              .format(model.score(X_train, y_train)))

        # exit()
        return model.predict(data_test)