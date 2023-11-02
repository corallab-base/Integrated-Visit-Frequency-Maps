import matplotlib.pyplot as plt
import numpy as np
import sys

_eps = 200

#####################################################################
# Defination of class

class result():
    def __init__(self, path, agents):
        self.agents = agents
        self.file = np.load(path, allow_pickle=True).squeeze()
        self.fail = 0
        self.rer_list, self.ce_list, self.pe_list, self.cmd_list, self.bandwidth_list, self.coverage_list, self.overlap_list, self.bandwidth_fast_list =\
              [], [], [], [], [], [], [], []

        for this_episode in self.file:            
            cmd_count = len(this_episode) // int(agents)
            # print(cmd_count)
            
            # print('end condition', bf[-1]['end_episode_condition'])
            if cmd_count == 0 or this_episode[-1]['end_episode_condition'] == 'timed_out':
                self.fail += 1
                continue

            for j in range(int(agents)):
                data = this_episode[-(j + 1)]
                # calculate rer and ce
                this_rer = data['repetitive_exploration_rate'] - 1
                # this_ce = float(data['explored_area']) / cmd_count
                this_coverage = data['ratio_explored']
                this_overlap = data['overlapped_ratio']

                # RER
                self.rer_list.append(this_rer)
                # CE
                # self.ce_list.append(this_ce)
                # PE
                if data['cumulative_distance'] >= 1:
                    this_pe = float(data['explored_area']) / data['cumulative_distance']
                    self.pe_list.append(this_pe)
                else:
                    self.pe_list.append(0.0)
                # CMD
                self.cmd_list.append(cmd_count)

                #COVERAGE
                self.coverage_list.append(this_coverage)

                # overlap
                self.overlap_list.append(this_overlap)

            this_bandwidth = sum(step_info['bandwidth'] for step_info in this_episode) 
            this_bandwidth_fast = sum(step_info['bandwidth_fast'] for step_info in this_episode) 

            # Bandwidth Fast
            self.bandwidth_fast_list.append(this_bandwidth_fast)

            # BANDWIDTH

            self.bandwidth_list.append(this_bandwidth)

            # calculate fail rate
            # if this_episode[-1]['end_episode_condition'] == 'timed_out':
            #    self.fail += 1

            self.np_rer_list = np.asarray(self.rer_list)
            # self.np_ce_list = np.asarray(self.ce_list)
            self.np_pe_list = np.asarray(self.pe_list)
            self.np_cmd_list = np.asarray(self.cmd_list)
            self.np_bandwidth_list = np.asarray(self.bandwidth_list)
            self.np_coverage_list = np.asarray(self.coverage_list)
            self.np_overlap_list = np.asarray(self.overlap_list)
            self.np_bandwidth_fast_list = np.asarray(self.bandwidth_fast_list)
    
    def print_stats(self):

        pm = '$\\pm$'

        # print('RER | PE | Steps | Overlap | Bandwidth | Coverage | Not Found')

        print(f'{np.nanmean(self.np_rer_list):.3f} {pm} {np.nanstd(self.np_rer_list):.3f}', end=' & ')

        print(f'{np.nanmean(self.np_pe_list):.0f} {pm} {np.std(self.np_pe_list):.0f}', end=' & ')

        print(f'{np.nanmean(self.np_cmd_list * int(self.agents)):.1f} {pm} {np.std(self.np_cmd_list * int(self.agents)):.1f}', end=' & ')

        if int(self.agents) == 1:
            print('N/A & ', end='')
        else:
            print(f'{np.nanmean(self.np_overlap_list):.1f} {pm} {np.std(self.np_overlap_list):.1f}', end=' & ')

        self.np_bandwidth_list_mib = np.array([b / 1024 / 1024 for b in self.np_bandwidth_fast_list])
        print(f'{np.nanmean(self.np_bandwidth_list_mib):.1f} {pm} {np.std(self.np_bandwidth_list_mib):.1f}', end=' & ')

        print(f'{np.nanmean(self.np_coverage_list):.3f} {pm} {np.std(self.np_coverage_list):.3f}', end=' & ')

        print(self.fail, end='')

        print(' \\\\', end='\n', flush=True)

#####################################################################
# Create results
def visualize(eval_path, num_agents):
    res = result(eval_path, num_agents)
    res.print_stats()

if __name__ == '__main__':
    visualize(sys.argv[2], sys.argv[1])

#####################################################################
# Make the plot

# # color list
# color_list = ['b-', 'g-', 'r-', 'y-', 'co', 'mo']

# # create x axis
# x_axis = range(_eps)
# fig, axs = plt.subplots(3, 1)

# # Upper image
# for i in range(len(result_list)):
#     res = result_list[i]
#     axs[0].plot(x_axis, res.np_rer_lst, color_list[i], label=res.name)
# axs[0].set(ylabel = 'GRER')
# axs[0].set_title('The GRERs of SAM-VFM, SAM, and ST-COM over 200 Testing Episodes')
# axs[0].legend()

# # Lower image
# for i in range(len(result_list)):
#     res = result_list[i]
#     axs[1].plot(x_axis, res.np_ce_lst, color_list[i], label=res.name)
# axs[1].set(ylabel = 'CE')
# axs[1].set_title('The GEs of SAM-VFM, SAM, and ST-COM over 200 Testing Episodes')
# #axs[1].legend()

# # Lower image
# for i in range(len(result_list)):
#     res = result_list[i]
#     axs[2].plot(x_axis, res.np_pe_lst, color_list[i], label=res.name)
# axs[2].set(xlabel='episodes', ylabel = 'PE')
# axs[2].set_title('The PEs of SAM-VFM, SAM, and ST-COM over 200 Testing Episodes')
# #axs[2].legend()

# plt.show()
