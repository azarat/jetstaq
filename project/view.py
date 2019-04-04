import os
import csv
import json
from operator import itemgetter
import numpy as np
from scipy import signal as signal_processor

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm

import project.controller as dt
import project.modules.q_profile as QProfileModule


class ViewData:
    internal_shots = 0
    internal_psi_r = 0
    internal_psi = 0
    internal_q_database = 0
    internal_q_x_database = 0
    internal_surfaces_ro = 0
    internal_surfaces_psinorm = 0
    internal_input_parametersa = 0

    processing = dt.PreProfiling()
    window_width_val_inv = 500  # time smoothing to find inv radius
    window_func = 'triang'
    close_plots = 1

    def __init__(self):
        print("--------------------------------------VERSION: 1.1")

        self.input_parameters = self.read_input_parameters()

        self.window_width_val_inv = self.input_parameters['advanced']['time_smooth_for_rinv']
        self.close_plots = self.input_parameters['advanced']['close_plots']
        self.window_func = self.input_parameters['advanced']['window_filtration_function']
        median_filter_window_size = (self.input_parameters['advanced']['median_filter_window_size'],
                                     self.input_parameters['advanced']['median_filter_window_size'])
        """ !OUTDATED! """
        # if self.close_plots == 0:
        #     """ Single dis. Offset for DB numbering """
        #     dis_end = self.input_parameters['required']['crashes']['start']
        #     dis_start = dis_end - 1
        # else:
        #     """ Range of dis """
        #     dis_start = self.input_parameters['required']['crashes']['start']
        #     dis_end = self.input_parameters['required']['crashes']['end'] + 1

        dis_start = self.input_parameters['required']['crashes']['start'] - 1
        dis_end = self.input_parameters['required']['crashes']['end']

        """ Outside loop """
        results = []
        db_controller = dt.Controller()
        db = db_controller.load()

        if type(self.shots) is int:
            self.shots = db_controller.shots

        for dis in range(dis_start, dis_end):

            print("------Load public data------DISCHARGE: ", self.shots[dis])
            data_public = self.prepare_data(database=db, source="public", dis=dis,
                                            median_filter_window_size=median_filter_window_size,
                                            window_filter_width_size=0, window_function=self.window_func)

            print("------Load real data------")
            data = self.prepare_data(database=db, source="real", dis=dis,
                                     median_filter_window_size=median_filter_window_size,
                                     window_filter_width_size=0, window_function=self.window_func)

            """ After r_inv, low accuracy, which have influence on r_inv detection """
            print("------Inversion radius detection------")
            temperature_matrix = np.asarray([channel[2] for channel in data])[:80, 9: -9]

            print('Smoothing channels along timeline')
            temperature_matrix_smooth = self.processing.filter(
                temperature_matrix, self.window_width_val_inv, self.window_func)

            print('R_inv detecting')
            r_maj_list = np.asarray([channel[1] for channel in data])[:len(temperature_matrix_smooth)]
            inv_radius_channel = dt.FindInvRadius().inv_radius(temperature_list=temperature_matrix_smooth,
                                                               window_width=6, std_low_limit=0.01,
                                                               channel_offset=15)

            inv_radius = 0
            if inv_radius_channel > 0:
                inv_radius = {
                    'index': np.asarray([channel[0] for channel in data])[inv_radius_channel],
                    'sorted_order': inv_radius_channel,
                    'position': '{:.4f}'.format(r_maj_list[inv_radius_channel]),
                    'position_neighbors': (r_maj_list[inv_radius_channel - 1], r_maj_list[inv_radius_channel + 1])
                }

                print(' ')
                print("Inversion radius index: " + str(inv_radius['index']))
                print("Inversion radius order number: " + str(inv_radius['sorted_order']))
                print("Inversion radius position: " + str(inv_radius['position']))

                print('Plotting results and save as images .PNG')
                self.build_temperature_rmaj_series_plot(temperature_matrix, self.window_width_val_inv,
                                                        r_maj_list, 1, discharge=dis, r_inv=inv_radius)

            else:
                print("FAILED")

            """ Identifying collapse duration """
            print("------Identifying collapse duration------")
            collapse_duration_time = dt.FindCollapseDuration().collapse_duration([],
                                                                                 temperature_matrix, 6,
                                                                                 inv_radius_channel, 1.03,
                                                                                 median_filter_window_size)

            time_list = data[0][3]

            collapse_duration_time = [int(x) for x in collapse_duration_time]
            print("Time segment: ", time_list[collapse_duration_time[0]], " ", time_list[collapse_duration_time[1]],
                  "ms | ", collapse_duration_time[0], " ", collapse_duration_time[1], " point numbers")
            print("Time duration: ",
                  (time_list[collapse_duration_time[1]] - time_list[collapse_duration_time[0]]) * 1000,
                  " ms")

            print('------Plotting results and save as images .PNG------')
            self.build_temperature_rmaj_single_plot(
                temperature_matrix,
                time_list,
                1, median_filter_window_size[0],
                time_limits=collapse_duration_time, discharge=dis, inv_radius_channel=inv_radius_channel)

            print("--------------------")

            if collapse_duration_time[0] == 0 or \
                    collapse_duration_time[1] == len(temperature_matrix) or \
                    len(temperature_matrix) == 0:
                print("FAILED")

            """ !IN DEV! """
            """ Q profile """
            print("")
            print("------------Start module: Q profile")

            """ Assign input parameters """
            q_profile = QProfileModule.QProfileModuleClass(self.input_parameters)

            """ Integrate and find required parameters """
            mix_pos = q_profile.get_x_mix(data, data_public, inv_radius, collapse_duration_time)
            r_mix = mix_pos["r_mix"]
            r_s = mix_pos["r_s"]
            qmix_to_qcenter = mix_pos["qmix_to_qcenter"]

            """ Transform R to PSI_norm surfaces_ro"""
            psi_xs = q_profile.get_psi_rmix(self.surfaces_ro, self.surfaces_psinorm, r_s)
            psi_rmix = q_profile.get_psi_rmix(self.surfaces_ro, self.surfaces_psinorm, r_mix)

            """ Find Q in defined PSI positions """
            q_rmix = q_profile.get_q_rmix(psi_rmix, self.q_database, self.surfaces_psinorm)
            q_center = 1 / (((-1 * qmix_to_qcenter) / q_rmix) + qmix_to_qcenter + 1)

            """ DEBUG """
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(15, 7)
            plt.close('all')
            ax.plot(self.surfaces_psinorm, self.q_database)

            x = [0, psi_xs, psi_rmix, self.surfaces_psinorm[-1]]
            y = [q_center, 1, q_rmix, self.q_database[-1]]
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(15, 7)
            ax.set(xlabel='PSI', ylabel='Q', title="QPROFILE, Discharge: " + str(self.shots[dis]) +
                                                   ", Shot: " + str(dis+1) +
                                                   " \n Center: " +
                                                   str(self.input_parameters['modules']['q_profile']['center']) +
                                                   "m, R_inv: " + str(inv_radius['position']) +
                                                   "m, Alpha: " + str(self.input_parameters['modules']['q_profile']['alpha']) +
                                                   ", Q_center: " + str('{:.4f}'.format(y[0])))
            ax.grid()
            ax.plot(x, y, 'o', x, y, '-')

            directory = 'results/modules/q_profile/'

            if not os.path.exists(directory):
                os.makedirs(directory)

            fig.savefig(directory + 'dis' + str(self.shots[dis]) +
                        '_crash' + str(dis + 1) +
                        '.png')
            plt.show()
            print("------End module: Q profile------")
            exit()
            """ ------!IN DEV! """

            result = [self.shots[dis],
                      time_list[collapse_duration_time[0]],
                      time_list[collapse_duration_time[1]],
                      (time_list[collapse_duration_time[1]] - time_list[collapse_duration_time[0]]) * 1000,
                      inv_radius['index'],
                      inv_radius['position']]

            results.append(result)

            print("--------------------------------------COMPLETE")
            print("\n\n\n")

        print('------Write results into file------')
        if self.close_plots == 0:
            plt.show()
        else:
            self.write_into_file(results)
        # # # # # # # # # # # # # # # # # # # # # # # #

    @staticmethod
    def read_input_parameters():
        """ -----------------------------------------
            version: 0.10
            desc: read input parameters from JSON file
            :return [list]
        ----------------------------------------- """

        with open('input.json') as f:
            input_parameters = json.load(f)

        return input_parameters

    def prepare_data(self, database, source, dis, median_filter_window_size, window_filter_width_size, window_function):
        """ -----------------------------------------
            version: 0.9
            desc: extract data from MatLab database, clear from channels with incomplete info,
                  sort channels, normalized on 1, smooth with different methods and combine into one object
            :return if source real: array with full info about each channel [int, float, array of float]
                    if source public: 2D array of floats
        ----------------------------------------- """

        """ Extract data from MATLAB database """
        print('Load data')
        data = dt.Profiling()
        data.assign(database=database, discharge=dis, source=source)

        if type(self.psi_r) is int:
            self.psi_r = data.psi_r

        if type(self.psi) is int:
            self.psi = data.psi

        if type(self.q_database) is int:
            self.q_database = data.q_database

        if type(self.q_x_database) is int:
            self.q_x_database = data.q_x_database

        if type(self.surfaces_ro) is int:
            self.surfaces_ro = data.surfaces_ro

        if type(self.surfaces_psinorm) is int:
            self.surfaces_psinorm = data.surfaces_psinorm

        if source == 'public':

            temperature_matrix = data.temperature_original
            return temperature_matrix

        else:

            """ Remove all channels with R_maj = nan """
            print('Remove channels with R_maj = nan')
            r_maj_list = self.processing.dict_to_list(data.channels_pos)
            temperature_matrix = data.temperature_original
            time_list = data.time

            temperature_matrix_buffer = []
            chan_order_buffer = []
            channels = {}
            for t_list_i, t_list in enumerate(temperature_matrix.items()):

                if r_maj_list[t_list_i] > 0:
                    """ temperature list of channel without r_maj = nan """
                    temperature_matrix_buffer.append(t_list[1])

                    """ channel number without r_maj = nan """
                    chan_order_buffer.append(t_list[0])

                    """ channel position (r_maj) without r_maj = nan (assigned to channel number) """
                    channels[t_list[0]] = r_maj_list[t_list_i]

            """ matrix without r_maj = nan """
            temperature_matrix = temperature_matrix_buffer
            """ transform back from list to dict """
            temperature_matrix = {chan_order_buffer[i]: k for i, k in enumerate(temperature_matrix)}
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ Sort channel number, channel position, channel temperature by R_maj value """
            print('Sort channels by their own R_maj value')
            channels_sorted = sorted(channels.items(), key=itemgetter(1))

            channel_number = [channel[0] for channel in channels_sorted]
            channel_position = [channel[1] for channel in channels_sorted]
            temperature_matrix = [temperature_matrix[channel[0]] for channel in channels_sorted]

            """
            Filtering T(t), i.e., smoothing
            WARNING: much info losses
            """
            # if window_filter_width_size > 0:
            #     print('Smoothing channels along timeline')
            #     temperature_matrix = self.processing.filter(
            #         temperature_matrix, window_filter_width_size, window_function)
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ Calibrate (Normalization on 1) """
            print('Normalizing channels on 1')
            temperature_matrix = data.normalization(temperature_matrix)
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ Median filtering. IMPORTANT to remove outliers """
            if median_filter_window_size != 0:
                temperature_matrix = signal_processor.medfilt2d(temperature_matrix, median_filter_window_size)

            """ Combine all data to one object """
            data = [(channel_number[i], channel_position[i], temperature_matrix[i], time_list)
                    for i, c in enumerate(temperature_matrix)]

            return data

    @staticmethod
    def write_into_file(results):
        """ -----------------------------------------
            version: 0.10
            desc: write results in file
            :return 1
        ----------------------------------------- """
        directory = 'results/'

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(directory + 'output.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Discharge order', 'Discharge JET number', 'Start, ms', 'End, ms', 'Duration, ms',
                                'Inv. radius channel', 'Inv. radius, m'])
            for row_i, row in enumerate(results):
                row = [row_i + 1] + row
                csvwriter.writerow(row)

        return 1

    def build_temperature_rmaj_single_plot(self, temperature, time_list, highlight_r_inv,
                                           start_offset, **kwargs):
        """ -----------------------------------------
            version: 0.3
        -----------------------------------------
        desc: Build single plot T(t) with fixed r_maj
        CAUTION: inverting plasma radius is near 48/49 channel (in ordered unit list)
        CAUTION: channel_to_check means temperature set, ordered by r_maj
        """

        fig, axes = plt.subplots()
        fig.set_size_inches(15, 7)

        temperature = temperature[:kwargs['inv_radius_channel']+10]

        for t_list_index, t_list in enumerate(temperature):
            if t_list_index % 4 == 0:
                axes.plot(
                    time_list[:len(t_list)],
                    t_list,
                    color="b"
                )

        """ Time limits of collapse """
        max_temp = np.amax(temperature)
        min_temp = np.amin(temperature)

        collapse_duration_txt = 0
        if kwargs['time_limits'] != 0 and kwargs['time_limits'][0] > 0 and kwargs['time_limits'][1] < len(time_list):
            if highlight_r_inv == 1:
                rect = patches.Rectangle((time_list[kwargs['time_limits'][0]], min_temp),
                                         (time_list[kwargs['time_limits'][1]] - time_list[kwargs['time_limits'][0]]),
                                         max_temp - min_temp, linewidth=0, edgecolor='r', facecolor='r', alpha=0.3)

                axes.add_patch(rect)

            collapse_duration_txt = '{:.4f}'.format((time_list[kwargs['time_limits'][1]] -
                                                     time_list[kwargs['time_limits'][0]]) * 1000) if kwargs['time_limits'] != 0 else 0

        axes.set(xlabel='Time (seconds)', ylabel='T (a.u.)',
                 title='Channels series, Discharge '
                        + str(self.shots[kwargs['discharge']]) +
                        str(', Crash ') + str(kwargs['discharge'] + 1) +
                        '\nCollapse duration = ' + str(collapse_duration_txt) + "ms")


        for item in ([axes.title, axes.xaxis.label, axes.yaxis.label] +
                     axes.get_xticklabels() + axes.get_yticklabels()):
            item.set_fontsize(17)

        axes.grid()

        directories = [
            'results/T_time/'
        ]

        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

            fig.savefig(directory + 'dis' + str(self.shots[kwargs['discharge']]) +
                        '_crash' + str(kwargs['discharge'] + 1) +
                        '_highlight' + str(highlight_r_inv) +
                        '.png')

        return 1

    def build_temperature_rmaj_series_plot(self, temperature_list, window_width, r_maj, highlight_r_inv, **kwargs):
        """ -----------------------------------------
            version: 0.3
            :return 1
        -----------------------------------------
        desc: Build multiple plots T(r_maj)
        with various fixed time
        """

        fig = plt.figure()
        axes = fig.add_subplot(111)
        fig.set_size_inches(15, 7)

        """ Fix determining minmax """
        plot_limit = len(temperature_list[0])
        temperature_list_buffer = []
        for time, temperature in enumerate(np.transpose(temperature_list)):
            if time in range(0, plot_limit, 100):
                temperature_list_buffer.append(temperature)

        label_limit = len(temperature_list)
        for time, temperature in enumerate(np.transpose(temperature_list)):
            """
            Labeling each channel (in ordered range)
            for T(R_maj) plot on the very beginning instant
            """
            if time == 0:

                """ Create double grid """
                axes.minorticks_on()
                axes.grid(which='minor', alpha=0.2)
                axes.grid(which='major', alpha=0.5)

                """ Create labels for every point on plot """
                # labels = range(label_limit)
                #
                # # order_ticks = []
                # for label, x, y in zip(labels, r_maj, temperature):
                #
                #     pos_offset = (0, 20)
                #     if label in range(0, label_limit):
                #         # order_ticks.append(x)
                #         plt.annotate(
                #             label,
                #             xy=(x, y), xytext=pos_offset,
                #             textcoords='offset points', ha='center', va='center',
                #             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5'))
                #
                # # ax2 = axes.twiny()
                # # ax2.set_xlim(axes.get_xlim())
                # # ax2.set_xticks(order_ticks)
                # # ax2.set_xticklabels(order_ticks)

                if highlight_r_inv == 1 and kwargs['r_inv']['index'] != 0:
                    rect = patches.Rectangle((kwargs['r_inv']['position_neighbors'][0], min(map(min, temperature_list_buffer))),
                                             (kwargs['r_inv']['position_neighbors'][1] - kwargs['r_inv']['position_neighbors'][0]),
                                             max(map(max, temperature_list_buffer)) - min(map(min, temperature_list_buffer)),
                                             linewidth=3, edgecolor='r', facecolor='r', alpha=0.2)
                    axes.add_patch(rect)

            """ Plot all temperature sets T(r_maj) """
            if time in range(0, plot_limit, 100):
                axes.plot(
                    r_maj,
                    temperature
                )

        # # # # # # # # # # # # # # # # # # # # # # # #

        # axes.set_xlim(min(r_maj), max(r_maj))
        axes.set(ylabel='T (a.u.)', xlabel='R (m)',
                 title= 'Discharge '
                        + str(self.shots[kwargs['discharge']]) +
                        str(', Crash ') + str(kwargs['discharge'] + 1) +
                       '\nR_inv: channel = ' + str(kwargs['r_inv']['index']) +
                       ', position = ' + str(kwargs['r_inv']['position']) + 'm')

        for item in ([axes.title, axes.xaxis.label, axes.yaxis.label] +
                     axes.get_xticklabels() + axes.get_yticklabels()):
            item.set_fontsize(17)

        directory = 'results/T_Rmaj/'

        if not os.path.exists(directory):
            os.makedirs(directory)

        fig.savefig(directory + 'dis' + str(self.shots[kwargs['discharge']]) +
                    '_crash' + str(kwargs['discharge'] + 1) +
                    '_T_Rmaj_series' +
                    '.png')

        return 1

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
    def input_parameters(self):
        return self.internal_input_parameters

    @input_parameters.setter
    def input_parameters(self, value):
        self.internal_input_parameters = value
