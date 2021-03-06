from Environment.Env import RealExpEnv
from Environment.get_atom_coordinate import get_atom_coordinate_nm, get_all_atom_coordinate_nm, get_atom_coordinate_nm_with_anchor

import numpy as np
import scipy.spatial as spatial
from scipy.optimize import linear_sum_assignment

def circle(x, y, r, p = 100):
    x_, y_ = [], []
    for i in range(p):
        x_.append(x+r*np.cos(2*i*np.pi/p))
        y_.append(y+r*np.sin(2*i*np.pi/p))
    return x_, y_ 

def assignment(start, goal):
    cost_matrix = spatial.distance.cdist(np.array(start)[:,:2], np.array(goal)[:,:2])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cost = cost_matrix[row_ind, col_ind]
    total_cost = np.sum(cost)
    return np.array(start)[row_ind,:], np.array(goal)[col_ind,:], cost, total_cost

class Structure_Builder(RealExpEnv):
    def __init__(self, step_nm, max_mvolt, max_pcurrent_to_mvolt_ratio, goal_nm, current_jump, im_size_nm, offset_nm,
                 pixel, scan_mV, max_len, correct_drift = False):
        super(Structure_Builder, self).__init__(step_nm, max_mvolt, max_pcurrent_to_mvolt_ratio, goal_nm, None, current_jump, im_size_nm, offset_nm,
                 None, pixel, None, None, scan_mV, max_len, correct_drift = False)
        self.atom_absolute_nm_f = None
        self.atom_absolute_nm_b = None
        self.large_DX_DDeltaX = float(self.createc_controller.stm.getparam('DX/DDeltaX'))

    def reset(self, destination_nm, anchor_nm, offset_nm, len_nm, large_len_nm):
        self.len = 0
        self.atom_absolute_nm, self.anchor_nm = self.scan_atom(anchor_nm, offset_nm, len_nm, large_len_nm)

        self.atom_start_absolute_nm = self.atom_absolute_nm
        destination_nm_with_correction = destination_nm + self.anchor_nm - anchor_nm
        self.destination_absolute_nm, self.goal = self.get_destination(self.atom_start_absolute_nm, destination_nm_with_correction)

        info = {'start_absolute_nm':self.atom_start_absolute_nm, 'goal_absolute_nm':self.destination_absolute_nm,
                'start_absolute_nm_f':self.atom_absolute_nm_f, 'start_absolute_nm_b':self.atom_absolute_nm_b, 'img_info':self.img_info}
        return np.concatenate((self.goal, (self.atom_absolute_nm - self.atom_start_absolute_nm)/self.goal_nm)), info
    def step(self, action):
        x_start_nm , y_start_nm, x_end_nm, y_end_nm, mvolt, pcurrent = self.action_to_latman_input(action)
        print(x_start_nm , y_start_nm, x_end_nm, y_end_nm, mvolt, pcurrent)
        current_series, d = self.step_latman(x_start_nm , y_start_nm, x_end_nm, y_end_nm, mvolt, pcurrent)
        info = {'current_series':current_series}
        info['d'] = d
        info['start_nm'] = np.array([x_start_nm , y_start_nm])
        info['end_nm'] = np.array([x_end_nm , y_end_nm])

        done = False
        self.len+=1

        if self.len == self.max_len:
            done = True
            self.dist_destination, dist_start, self.cos_similarity_destination = self.check_similarity()
        else:
            jump = self.detect_current_jump(current_series)
            if jump:
                self.dist_destination, dist_start, self.cos_similarity_destination = self.check_similarity()
                print('atom moves by:', dist_start)
                if (dist_start/self.goal_nm) > 1.5 or self.dist_destination < 0.16:
                    done = True

        next_state = np.concatenate((self.goal, (self.atom_absolute_nm - self.atom_start_absolute_nm)/self.goal_nm))

        info['atom_absolute_nm'] = self.atom_absolute_nm
        info['atom_absolute_nm_f'] = self.atom_absolute_nm_f
        info['atom_absolute_nm_b'] = self.atom_absolute_nm_b
        info['img_info'] = self.img_info
        return next_state, None, done, info

    def scan_atom(self, anchor_nm = None, offset_nm = None, len_nm = None, large_len_nm = None):
        if offset_nm is None:
            offset_nm = self.offset_nm
        if len_nm is None:
            len_nm = self.len_nm
        if anchor_nm is None:
            anchor_nm = self.anchor_nm
        if large_len_nm is not None:
            small_DX_DDeltaX = int(self.large_DX_DDeltaX*len_nm/large_len_nm)
            self.createc_controller.stm.setparam('DX/DDeltaX', small_DX_DDeltaX)

        self.createc_controller.offset_nm = offset_nm
        self.createc_controller.im_size_nm = len_nm
        self.offset_nm = offset_nm
        self.len_nm = len_nm
        self.anchor_nm = anchor_nm
        img_forward, img_backward, offset_nm, len_nm = self.createc_controller.scan_image()
        self.img_info = {'img_forward':img_forward,'img_backward':img_backward, 'offset_nm':offset_nm, 'len_nm':len_nm}
        self.atom_absolute_nm_f, self.anchor_nm_f = get_atom_coordinate_nm_with_anchor(img_forward, offset_nm, len_nm, anchor_nm)

        self.atom_absolute_nm_b, self.anchor_nm_b = get_atom_coordinate_nm_with_anchor(img_backward, offset_nm, len_nm, anchor_nm)
        self.atom_absolute_nm = 0.5*(self.atom_absolute_nm_f+self.atom_absolute_nm_b)
        self.anchor_nm = 0.5*(self.anchor_nm_f+self.anchor_nm_b)
        return self.atom_absolute_nm, self.anchor_nm

    def scan_all_atoms(self, offset_nm, len_nm):
        self.createc_controller.stm.setparam('DX/DDeltaX', self.large_DX_DDeltaX)
        self.createc_controller.offset_nm = offset_nm
        self.createc_controller.im_size_nm = len_nm
        self.offset_nm = offset_nm
        self.len_nm = len_nm
        img_forward, img_backward, offset_nm, len_nm = self.createc_controller.scan_image()
        all_atom_absolute_nm_f = get_all_atom_coordinate_nm(img_forward, offset_nm, len_nm)
        all_atom_absolute_nm_b = get_all_atom_coordinate_nm(img_backward, offset_nm, len_nm)

        all_atom_absolute_nm_f = np.array(sorted(all_atom_absolute_nm_f, key = lambda x: x[0]))
        all_atom_absolute_nm_f = np.array(sorted(all_atom_absolute_nm_f, key = lambda x: x[1]))

        all_atom_absolute_nm_b = np.array(sorted(all_atom_absolute_nm_b, key = lambda x: x[0]))
        all_atom_absolute_nm_b = np.array(sorted(all_atom_absolute_nm_b, key = lambda x: x[1]))

        self.all_atom_absolute_nm_f = all_atom_absolute_nm_f
        self.all_atom_absolute_nm_b = all_atom_absolute_nm_b

        if len(all_atom_absolute_nm_b)!=len(all_atom_absolute_nm_f):
            print('length of list of atoms found in b and f different')
         #   all_atom_absolute_nm_b,all_atom_absolute_nm_f = self.remove_outliers(all_atom_absolute_nm_b,all_atom_absolute_nm_f)

        all_atom_absolute_nm = 0.5*(all_atom_absolute_nm_f+all_atom_absolute_nm_b)
        self.all_atom_absolute_nm = all_atom_absolute_nm
        return all_atom_absolute_nm, img_forward, img_backward, offset_nm, len_nm

    def get_destination(self, atom_start_absolute_nm, destination_absolute_nm):
        angle = np.arctan2((destination_absolute_nm-atom_start_absolute_nm)[1],(destination_absolute_nm-atom_start_absolute_nm)[0])
        goal_nm = min(self.goal_nm, np.linalg.norm(destination_absolute_nm-atom_start_absolute_nm))
        destination_absolute_nm = atom_start_absolute_nm + goal_nm*np.array([np.cos(angle),np.sin(angle)])
        return destination_absolute_nm, goal_nm*np.array([np.cos(angle),np.sin(angle)]/self.goal_nm)

    def step_latman(self, x_start_nm, y_start_nm, x_end_nm, y_end_nm, mvoltage, pcurrent):
        #pdb.set_trace()
        #print(x_start_nm, y_start_nm, x_end_nm, y_end_nm, mvoltage, pcurrent)
        x_start_nm+=self.atom_absolute_nm[0]
        x_end_nm+=self.atom_absolute_nm[0]
        y_start_nm+=self.atom_absolute_nm[1]
        y_end_nm+=self.atom_absolute_nm[1]
        if [x_start_nm, y_start_nm] != [x_end_nm, y_end_nm]:
            #print(x_start_nm, y_start_nm, x_end_nm, y_end_nm, mvoltage, pcurrent, self.offset_nm, self.len_nm)
            data = self.createc_controller.lat_manipulation(x_start_nm, y_start_nm, x_end_nm, y_end_nm, mvoltage, pcurrent, self.offset_nm, self.len_nm)
            if data is not None:
                current = np.array(data.current).flatten()
                x = np.array(data.x)
                y = np.array(data.y)
                d = np.sqrt(((x-x[0])**2 + (y-y[0])**2))
            else:
                current = None
                d = None
            return current, d
        else:
            return None, None
