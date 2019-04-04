import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as signal_processor


class QProfileModuleClass:
    center = 0

    internal_channel_index = 0
    internal_channel_position = 0
    internal_input_parameters = 0

    def __init__(self, input_parameters):
        """ -----------------------------------------
            version: 0.10
            desc: assign parameters
        ----------------------------------------- """

        self.input_parameters = input_parameters['modules']['q_profile']
        self.center = self.input_parameters['center']

    def get_x_mix(self, data, data_public, inv_radius, collapse_duration_time):
        """ -----------------------------------------
            version: 1.1
            desc: 
            :return
        ----------------------------------------- """

        channel_index = np.asarray([channel[0] for channel in data])
        channel_position = np.asarray([channel[1] for channel in data])
        temperature_matrix = np.asarray([channel[2] for channel in data])
        temperature_matrix_public = data_public

        """ !OUTDATED! """
        # xc_index = xc_index_order['index']
        # xs = self.find_xs(temperature_matrix_public, temperature_matrix, channel_position, channel_index, xc_index, inv_radius, collapse_duration_time[1])
        # xs = np.power((np.sqrt(xs) + self.center), 2)
        # xs_index_order = self.nearest_channel_index(np.sqrt(xs), channel_position, channel_index)

        """ Normalization to PPF """
        temperature_matrix = self.normalization_ppf(temperature_matrix, temperature_matrix_public, channel_index)

        temperature_pre_post = {
            'pre': np.transpose(temperature_matrix)[collapse_duration_time[0]],
            'post': np.transpose(temperature_matrix)[collapse_duration_time[1]]
        }

        """ Median det. killer (median filtration) """
        temperature_pre_post['pre'] = signal_processor.medfilt(temperature_pre_post['pre'], 3)
        temperature_pre_post['post'] = signal_processor.medfilt(temperature_pre_post['post'], 31)

        print("------Correction of value of magnet center------")
        temperature_pre = temperature_pre_post['pre']
        temperature_pre = temperature_pre.tolist()
        xc_order = temperature_pre.index(max(temperature_pre))

        old_center = self.center
        print("Old magnet center: ", old_center, "m")
        self.center = channel_position[xc_order]

        new_center = self.center
        print("New magnet center: ", new_center, "m")

        diffenrence_centers = (np.abs(new_center - old_center) / old_center) * 100
        print("Difference: ", ("%.2f" % diffenrence_centers), "%")

        """ Save original values """
        self.channel_index = channel_index
        self.channel_position = channel_position

        """ Normalizing channel position values on inversion radius """
        inv_radius_norm = np.power((float(inv_radius['position']) - self.center), 2)
        # channel_position_o = channel_position
        channel_position = [v-self.center for v in channel_position]
        channel_position = [((np.power(v, 2)) * (np.sign(v))) / inv_radius_norm for v in channel_position]

        xc_index_order = self.nearest_channel_index(0, channel_position, channel_index)

        """ Kill detalization (window filtration) """
        # window = signal_processor.get_window('triang', 10)
        # temperature_pre_post['pre'] = signal_processor.convolve(temperature_pre_post['pre'], window, mode='valid')
        # temperature_pre_post['post'] = signal_processor.convolve(temperature_pre_post['post'], window, mode='valid')

        xs = self.find_xs_ppf(temperature_pre_post, channel_position)
        xs_index_order = self.nearest_channel_index(xs, channel_position, channel_index)

        integral = self.integrate_eiler_linear_aprox(xs, xs_index_order, channel_position, temperature_pre_post, inv_radius, xc_index_order)

        x_mix = integral["x_2"][-1]
        # x_mix = 2.2
        r_mix = np.sqrt(x_mix * inv_radius_norm) + self.center
        r_s = np.sqrt(xs * inv_radius_norm) + self.center

        # """ DEBUG """
        # print(integral["x_2"])
        # print(xs)
        # print(x_mix)
        # exit()

        return {
            "r_s": r_s,
            "r_mix": r_mix,
            "qmix_to_qcenter": np.abs(integral["f"][-1])
        }

    def find_xs_ppf(self, temperature_pre_post, channel_position):

        temperature_xc = self.temperature_interpolation(0, channel_position, temperature_pre_post['post'])

        diff = [temperature_xc - v for v in reversed(temperature_pre_post['pre'])]

        channel_before = 0
        channel_after = 0
        for i, v in enumerate(diff):
            if v <= 0:
                channel_after = len(diff) - i
                channel_before = len(diff) - i - 1
                break

        position_gap = np.linspace(channel_position[channel_before], channel_position[channel_after], 100)
        temperature_gap = np.linspace(temperature_pre_post['pre'][channel_before], temperature_pre_post['pre'][channel_after], 100)

        diff_gap = [temperature_xc-v for v in temperature_gap]

        point_cs = 0

        for i, v in enumerate(diff_gap):
            if v >= 0:
                point_cs = i
                break

        position_cs = position_gap[point_cs]

        return position_cs

    @staticmethod
    def normalization_ppf(temperature_matrix, temperature_matrix_public, channel_index):
        """ -----------------------------------------
            version: 0.9
            desc:
            :return
        ----------------------------------------- """

        temperature_matrix_norm = []
        for order, index in enumerate(channel_index):
            temperature_matrix_norm.append(temperature_matrix[order] * temperature_matrix_public[index][1])

        return temperature_matrix_norm

    @staticmethod
    def temperature_interpolation(inter_position, channel_position, temperature):
        """ -----------------------------------------
            version: 0.9
        ----------------------------------------- """

        # channel_position = [np.power(v, 2) for v in channel_position]
        diff = [v-inter_position for v in channel_position]

        channel_after = 0
        channel_before = 0
        for i, v in enumerate(diff):
            if v >= 0:
                channel_before = i-1
                channel_after = i
                break

        """ Difference between siblings temperature"""
        diff_temperature = temperature[channel_after] - temperature[channel_before]

        if diff_temperature == 0:
            return temperature[channel_after]

        a = np.abs(diff_temperature)

        """ Difference between siblings position (distance) """
        b = channel_position[channel_after] - channel_position[channel_before]
        c = np.sqrt( np.power(a, 2) + np.power(b, 2) )

        """ Difference between interpoint and channel after (distance) """
        b2 = channel_position[channel_after] - inter_position
        """ Difference between interpoint and channel before (distance) """
        b1 = b - b2

        """ Hypotenuse between channel before and interpoint """
        c1 = b1 * c / b

        """ Difference between interpoint temperature and one of siblings temperature """
        h1 = np.sqrt( np.power(c1, 2) - np.power(b1, 2) )

        intertemperature = 0
        if diff_temperature <= 0:
            intertemperature = temperature[channel_before] - h1
        elif diff_temperature > 0:
            intertemperature = temperature[channel_before] + h1

        return intertemperature

    def find_xs(self, temperature_matrix_public, temperature_matrix, channel_position, channel_index, xc_index, inv_radius, crash_end):
        """ -----------------------------------------
            !OUDATED!
            version: 0.9
            desc: t => temperature
                  x = r^2
            :param temperature_matrix_public:
            :param temperature_matrix:
            :param channel_position:
            :param channel_index:
            :return: position of xs [float]
        ----------------------------------------- """

        """ convert ndarray to list """
        channel_index = [v for v in channel_index]

        t_center_precrash_norm = 1
        """ we should calculate xs from the magnet center (not from the center of torus) """
        x_inv = pow(float(inv_radius['position'])-self.center, 2)

        t_center_precrash_public = temperature_matrix_public[xc_index][3]
        xc_order = channel_index.index(xc_index)
        t_center_postcrash_norm = temperature_matrix[xc_order][crash_end]

        t_inv_public = temperature_matrix_public[inv_radius['index']][3]

        t_inv_norm = t_inv_public / t_center_precrash_public
        h = 1 - (t_inv_norm / t_center_postcrash_norm)
        g = h * (x_inv / t_center_precrash_norm)

        xs = x_inv - g

        return xs

    @staticmethod
    def nearest_channel_index(position, channel_position, channel_index):
        """ -----------------------------------------
            version: 0.9
            desc: find the index of the channel which is nearest to the position
            :param position:
            :param channel_position:
            :param channel_index:
            :return: index of nearest channel to the input position value [int]
        ----------------------------------------- """

        channel_position_diff = [abs(x - position) for x in channel_position]
        diff_min = min(channel_position_diff)
        position_channel_order = channel_position_diff.index(diff_min)

        channel_index_and_order = {
            'index': channel_index[position_channel_order],
            'order': position_channel_order
        }

        return channel_index_and_order

    def debug_density(self, x_1, x_2):
        """ -----------------------------------------
            version: 1.1
        ----------------------------------------- """

        alpha = self.input_parameters['alpha']

        """ DEBUG """
        # print((1 - alpha * x_1), " / ", (1 - alpha * x_2))
        # print(alpha)
        # print(x_1)
        # print(x_2)
        # print("")
        # print("")

        return_f = (1 - alpha * x_1) / (1 - alpha * x_2)

        return return_f

    def debug_interfunc(self, temperature_1, temperature_2, temperature_plus):
        """ -----------------------------------------
            version: 1.1
        ----------------------------------------- """

        return_f = (temperature_1 - temperature_plus) / (temperature_plus - temperature_2)

        return return_f

    def integration_function(self, temperature_plus, temperature_1, temperature_2, x_1, x_2):
        """ -----------------------------------------
            version: 1.1
        ----------------------------------------- """

        """ With electron density """
        # alpha = self.input_parameters['alpha']
        # return_f = -1 * ((1 - alpha * x_1) / (1 - alpha * x_2)) * ((temperature_1 - temperature_plus) / (temperature_plus - temperature_2))

        """ Without electron density """
        return_f = -1 * (temperature_1 - temperature_plus) / (temperature_plus - temperature_2)

        return return_f

    def integrate_eiler(self, xs, channel_position, temperature_pre_post):
        """ -----------------------------------------
            version: 1.1
        ----------------------------------------- """

        n_steps = 100
        integration_boundaries = np.linspace(xs, 0, n_steps)

        y = 0
        function_value = 0
        x_1 = []
        f = []
        T_1 = []
        T_2 = []
        T_plus = []
        x_plus = []
        x_2 = []
        for i, inter_position in enumerate(integration_boundaries):

            if i == 0:
                y = xs
                function_value = -1
            else:

                h = (inter_position - integration_boundaries[i-1])
                y = y + h * function_value

                temperature_1 = self.temperature_interpolation(inter_position, channel_position, temperature_pre_post['pre'])
                temperature_2 = self.temperature_interpolation(y, channel_position, temperature_pre_post['pre'])
                temperature_plus = self.temperature_interpolation((y - inter_position), channel_position, temperature_pre_post['post'])

                """ Temperature correction due to the overlapping T1/2 and T+ """
                temperature_xs = self.temperature_interpolation(xs, channel_position, temperature_pre_post['pre'])
                diff_Txs_Tx2 = np.abs(temperature_xs - temperature_2)
                diff_Txs_Txplus = np.abs(temperature_xs - temperature_plus)

                """ NEED TO BE TESTED """
                if (diff_Txs_Txplus / diff_Txs_Tx2) > 0.4:
                    temperature_plus = (diff_Txs_Tx2 / 1.5) + temperature_2
                """ NEED TO BE TESTED """

                function_value = self.integration_function(temperature_plus, temperature_1, temperature_2, inter_position, y)

                debug_density = self.debug_density(inter_position, y)

                if np.isnan(inter_position) or np.isnan(y - inter_position) or np.isnan(y) \
                        or np.isnan(temperature_1) or np.isnan(temperature_2) or np.isnan(temperature_plus) \
                        or debug_density < 0 or (len(x_2) > 0 and y < x_2[-1]):
                    break

                x_2.append(y)
                x_plus.append(y - inter_position)
                x_1.append(inter_position)
                f.append(function_value)
                T_1.append(temperature_1)
                T_2.append(temperature_2)
                T_plus.append(temperature_plus)

        return {
            "f": f,
            "x_2": x_2
        }

    def linearization(self, temperature, xs, channel_position):

        temperature_xc = self.temperature_interpolation(0, channel_position, temperature)
        temperature_xs = self.temperature_interpolation(xs, channel_position, temperature)

        # Linear extrapolation
        temperature_xlast = temperature_xc + ((channel_position[-1] - 0) / (xs - 0)) * (temperature_xs - temperature_xc)

        temperature_linear = np.linspace(temperature_xc, temperature_xlast, len(temperature))
        position_linear = np.linspace(channel_position[0], channel_position[-1], len(temperature))

        temperature_reconstructed = []
        for p in channel_position:
            temperature_reconstructed.append(self.temperature_interpolation(p, position_linear, temperature_linear))

        temperature_xs = self.temperature_interpolation(xs, channel_position, temperature_reconstructed)

        return temperature_reconstructed

    def integrate_eiler_linear_aprox(self, xs, xs_index_order, channel_position, temperature_pre_post, inv_radius, xc):
        """ -----------------------------------------
            version: 1.2
        ----------------------------------------- """

        temperature_pre_post_post = temperature_pre_post['post']

        channel_position_cut = []
        for p in channel_position:
            if p > 0:
                channel_position_cut.append(p)

        channel_position = channel_position_cut

        cut_negative = len(temperature_pre_post['pre']) - len(channel_position)
        temperature_pre_post['pre'] = temperature_pre_post['pre'][cut_negative:]
        temperature_pre_post['post'] = temperature_pre_post['post'][cut_negative:]
        temperature_pre_post_post = temperature_pre_post_post[cut_negative:]

        temperature_pre_post['post'] = self.linearization(temperature_pre_post['post'], xs-1, channel_position)

        n_steps = 1000
        integration_boundaries = np.linspace(xs, 0, n_steps)

        y = 0
        h = 0
        temperature_1 = 0
        temperature_2 = 0
        temperature_plus = 0
        function_value = 0
        x_1 = []
        f = []
        T_1 = []
        T_2 = []
        T_plus = []
        x_plus = []
        h_array = []
        x_2 = []
        debug_interfunc_array = []
        debug_density_array = []
        return_f_1 = []
        return_f_2 = []
        for i, inter_position in enumerate(integration_boundaries):

            if i == 0:
                y = xs
                function_value = -1
            else:

                h = (inter_position - integration_boundaries[i-1])
                y = y + h * function_value

                temperature_1 = self.temperature_interpolation(inter_position, channel_position, temperature_pre_post['pre'])
                temperature_2 = self.temperature_interpolation(y, channel_position, temperature_pre_post['pre'])
                temperature_plus = self.temperature_interpolation((y - inter_position), channel_position, temperature_pre_post['post'])

                """ Temperature correction due to the overlapping T1/2 and T+ """
                temperature_xs = self.temperature_interpolation(xs, channel_position, temperature_pre_post['pre'])
                diff_Txs_Tx2 = np.abs(temperature_xs - temperature_2)
                diff_Txs_Txplus = np.abs(temperature_xs - temperature_plus)

                """ NEED TO BE TESTED """
                if (diff_Txs_Txplus / diff_Txs_Tx2) > 0.4:
                    temperature_plus = (diff_Txs_Tx2 / 1.5) + temperature_2
                """ NEED TO BE TESTED """

                function_value = self.integration_function(temperature_plus, temperature_1, temperature_2, inter_position, y)

                debug_density = self.debug_density(inter_position, y)
                debug_interfunc = self.debug_interfunc(temperature_1, temperature_2, temperature_plus)

                # """ DEBUG """
                # print("-------------------------STEP ", i)
                # print("x1 = ", inter_position, " --- xs = ", xs, " --- y = ", y, " --- x+ = ", (y - inter_position))
                # if len(x_2) > 0:
                #     print("-------------------------")
                #     print("x2_step = ", y-x_2[-1])
                # print("-------------------------")
                # print("diff_Txs_Txplus / diff_Txs_Tx2 = ", diff_Txs_Txplus / diff_Txs_Tx2)
                # diff_Txs_Tx2 = np.abs(temperature_xs - temperature_2)
                # diff_Txs_Txplus = np.abs(temperature_xs - temperature_plus)
                # print("diff_Txs_Txplus / diff_Txs_Tx2 = ", diff_Txs_Txplus / diff_Txs_Tx2)
                # print("-------------------------")
                # print("T1 = ", temperature_1, " --- T2 = ", temperature_2, " --- T+ = ", temperature_plus)
                # print("-------------------------")
                # print("f = ", function_value)
                # print("-------------------------")

                if np.isnan(inter_position) or np.isnan(y - inter_position) or np.isnan(y) \
                        or np.isnan(temperature_1) or np.isnan(temperature_2) or np.isnan(temperature_plus) \
                        or debug_density < 0 or (len(x_2) > 0 and y < x_2[-1]):
                    break

                """ DEBUG """
                alpha = self.input_parameters['alpha']
                return_f_1.append(1 - alpha * inter_position)
                return_f_2.append(1 - alpha * y)
                debug_interfunc_array.append(debug_interfunc)
                debug_density_array.append(debug_density)
                h_array.append(h)

                x_2.append(y)
                x_plus.append(y - inter_position)
                x_1.append(inter_position)
                f.append(function_value)
                T_1.append(temperature_1)
                T_2.append(temperature_2)
                T_plus.append(temperature_plus)

        # """ DEBUG """
        # # xm = np.linspace(0, 1, 100)
        # # # tests = [(((-1 * np.power(x, 2) * 10) + 10)) for x in xm]
        # # tests = [(1 - (0.6 * x)) for x in xm]
        #
        # plt.close('all')
        # # fig, ax = plt.subplots(1, 1)
        # # fig.set_size_inches(15, 7)
        # # ax.plot(x_1)
        # # ax.plot(x_2)
        # # ax.plot(x_plus)
        # # ax.plot(integration_boundaries)
        # # f = [-1 * v for v in f]
        # # ax.plot(f)
        # # f = signal_processor.medfilt(f, 11)
        # # ax.plot(f)
        # # ax.plot(xm, tests)
        # # ax.plot(debug_interfunc_array)
        # # ax.plot(debug_density_array)
        # # ax.plot(return_f_1)
        # # ax.plot(return_f_2)
        # # ax.plot(channel_position, temperature_pre_post['post'])
        #
        # fig, ax = plt.subplots(1, 1)
        # fig.set_size_inches(15, 7)
        #
        # ax.set(xlabel='X', ylabel='T', title="Discharge: 86459")
        # ax.plot(channel_position, temperature_pre_post['pre'], "--")
        # ax.plot(channel_position, temperature_pre_post['post'], "--")
        # ax.plot(channel_position, temperature_pre_post_post, "--")
        #
        # # ax.axvline(x=channel_position[xc['order']], color="black")
        # ax.axvline(x=xs, color="black")
        # ax.axhline(y=temperature_pre_post['post'][cut_negative], color="black")
        # # ax.axvline(x=channel_position[inv_radius['sorted_order']], color="black")
        # ax.axvline(x=x_2[-1], color="red")
        # ax.axvline(x=x_1[-1], color="red")
        # # ax.axvline(x=x_plus[-1], color="orange")
        # # ax.axhline(y=T_plus[-1], color="orange")
        # # ax.legend(["T pre", "T post", "Xs", "T(Xs)", "X2"])
        #
        # for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
        #              ax.get_xticklabels() + ax.get_yticklabels()):
        #     item.set_fontsize(17)
        #
        # plt.show()
        # exit()
        # # f = signal_processor.medfilt(f, 11)

        return {
            "f": f,
            "x_2": x_2
        }

    def get_psi_rmix(self, psi_r, psi, r_mix):
        """ -----------------------------------------
             version: 0.10
             desc: find value psi_r in a point r_mix
                   psi_r ~ r^2
         ----------------------------------------- """

        diff_r = psi_r - r_mix

        left_channel_order = 0
        right_channel_order = 0
        for i, d in enumerate(diff_r):
            if d > 0:
                left_channel_order = i-1
                right_channel_order = i
                break

        """ Find interposition via triangular equations """
        """ Katet """
        b = psi_r[right_channel_order] - psi_r[left_channel_order]
        """ Part of the same katet """
        b1 = r_mix - psi_r[left_channel_order]
        """ Another katet """

        a = psi[right_channel_order] - psi[left_channel_order]
        """ Hypotenuza """
        c = np.sqrt(np.power(a, 2) + np.power(b, 2))
        """ Triangular part of psi we looking for """
        a1 = b1 * b / c

        if psi_r[left_channel_order] < psi_r[right_channel_order]:
            psi_rmix = psi[left_channel_order] + a1
        else:
            psi_rmix = psi[left_channel_order] - a1

        return psi_rmix

    def get_q_rmix(self, psi_rmix, q_database, psi_database):
        """ -----------------------------------------
             version: 1.1
             desc:
         ----------------------------------------- """

        # psi_database = psi_database.tolist()
        # min_psi_index = psi_database.index(min(psi_database))
        # psi_database = psi_database[min_psi_index:]
        # q_database = q_database[min_psi_index:]

        diff_psi = psi_database - psi_rmix

        left_channel_order = 0
        right_channel_order = 0
        for i, d in enumerate(diff_psi):
            if d > 0:
                left_channel_order = i-1
                right_channel_order = i
                break

        """ Find interposition via triangular equations """
        """ Katet """
        b = psi_database[right_channel_order] - psi_database[left_channel_order]
        """ Part of the same katet """
        b1 = psi_rmix - psi_database[left_channel_order]
        """ Another katet """

        a = q_database[right_channel_order] - q_database[left_channel_order]
        """ Hypotenuza """
        c = np.sqrt(np.power(a, 2) + np.power(b, 2))
        """ Triangular part of psi we looking for """
        a1 = b1 * b / c

        if q_database[left_channel_order] < q_database[right_channel_order]:
            q_rmix = q_database[left_channel_order] + a1
        else:
            q_rmix = q_database[left_channel_order] - a1

        return q_rmix

    @property
    def channel_index(self):
        return self.internal_channel_index

    @channel_index.setter
    def channel_index(self, value):
        self.internal_channel_index = value

    @property
    def input_parameters(self):
        return self.internal_input_parameters

    @input_parameters.setter
    def input_parameters(self, value):
        self.internal_input_parameters = value

    @property
    def channel_position(self):
        return self.internal_channel_position

    @channel_position.setter
    def channel_position(self, value):
        self.internal_channel_position = value
