from copy import deepcopy
import getopt
import random
from ArgumentValue import prob_bool
import z3
import sys
import os
import hashlib
import subprocess
import multiprocessing
import json
from Argument import Argument
from ArgType import ArgType
from APIManager import APIManager
from utils import *
from Runner import Runner
import numpy as np
import time
import re
import traceback
from Option import TestOption
import time

def all_smt(s, initial_terms):
    def block_term(s, m, t):
        if z3.is_array(t):
            v = m[t]
            v = v.children()
            while(len(v) == 3):
                idx = v[1].as_long()
                val = v[2].as_long()
                if idx > 20:
                    v = v[0].children()
                    continue
                v = v[0].children()
                s.add(t[idx] != m.eval(t[idx], model_completion=True))
        else:
            s.add(t != m.eval(t, model_completion=True))
    def fix_term(s, m, t):
        s.add(t == m.eval(t, model_completion=True))
    def all_smt_rec(terms):
        if z3.sat == s.check():
            m = s.model()
            yield m
            for i in range(len(terms)):
                s.push()
                block_term(s, m, terms[i])
                for j in range(i):
                    fix_term(s, m, terms[j])
                yield from all_smt_rec(terms[i:])
                s.pop()
    yield from all_smt_rec(list(initial_terms))
    yield None


BIT_WIDTH = 64
extend_expr_id = 0

def get_unique_extend_expr_name():
    global extend_expr_id
    extend_expr_id += 1
    return 'extend_' + str(extend_expr_id)


class MultiProcessHandler:
    def __init__(self, goals, sample_num=None, task='check', args=None, test_option=None):
        self.p_num = test_option.p_num
        self.goals = goals
        self.task = task
        self.sample_num = sample_num
        self.res = [ None for _ in self.goals ]
        self.args = args
        if len(self.goals) == 0:
            return
        if test_option is not None:
            self.sampler = test_option.sampler

    def execute(self):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        index = 0
        processes = []
        while len(processes) < self.p_num and index < len(self.goals):
            p = None
            if self.task == 'check':
                p = multiprocessing.Process(target=self.check, args=(index, self.goals[index], return_dict))
            elif self.task == 'sample':
                p = multiprocessing.Process(target=self.sample_z3, args=(index, self.goals[index], self.sample_num, return_dict, self.args))
                # p = multiprocessing.Process(target=self.sample_z3, args=(index, self.goals[index], self.sample_num, return_dict, self.sampler))
            elif self.task == 'extend':
                p = multiprocessing.Process(target=self.extend, args=(index, self.goals[index], self.args, return_dict))
            p.start()
            processes.append(p)
            index += 1

        while True:
            time.sleep(0.5)
            print('{}/{} finished.'.format(len(return_dict.keys()), len(self.goals)), end='\r', flush=True)
            all_finished = True
            for i in range(len(processes)):
                p = processes[i]
                if p is None:
                    continue
                # not finished yet
                if p.is_alive():
                    all_finished = False
                else:
                    if index < len(self.goals):
                        p = None
                        if self.task == 'check':
                            p = multiprocessing.Process(target=self.check, args=(index, self.goals[index], return_dict))
                        elif self.task == 'sample':
                            p = multiprocessing.Process(target=self.sample_z3, args=(index, self.goals[index], self.sample_num, return_dict, self.args))
                            # p = multiprocessing.Process(target=self.sample_z3, args=(index, self.goals[index], self.sample_num, return_dict, self.sampler))
                        elif self.task == 'extend':
                            p = multiprocessing.Process(target=self.extend, args=(index, self.goals[index], self.args, return_dict))
                        processes[i] = p
                        p.start()
                        index += 1
                        all_finished = False
                    else:
                        processes[i] = None
            if all_finished:
                break
        for i in range(len(self.goals)):
            if i in return_dict.keys():
                self.res[i] = return_dict[i]
        if self.task == 'sample':
            os.system('kill -9 $(pidof smtsampler)')

    @staticmethod
    def extend(res_id, goal, args, ret_dict):
        def get_arg(arg_name):
            for arg in args:
                if arg.name == arg_name:
                    return arg
            return None
        all_vars = get_vars_in_goal(goal)
        for var in all_vars:
            arg_name, var_type = VarType.get_var_type(str(var))
            if var_type == VarType.VAR:
                continue
            elif var_type == VarType.TENSOR_DIM:
                goal.add(var.n >= z3.BitVecVal(0, BIT_WIDTH))
                goal.add(var.n <= z3.BitVecVal(6, BIT_WIDTH))
            elif var_type == VarType.LIST_LEN:
                goal.add(var.n >= z3.BitVecVal(0, BIT_WIDTH))
                goal.add(var.n <= z3.BitVecVal(10, BIT_WIDTH))
            elif var_type == VarType.STR_FMT:
                if get_arg(arg_name) is None:
                    continue
                assert(len(get_arg(arg_name).arg_prop.enum) > 0)
                expr = z3.BoolVal(False)
                for fmt in get_arg(arg_name).arg_prop.enum:
                    expr = z3.Or(expr, var.n == z3.BitVecVal(fmt_to_int(fmt), BIT_WIDTH))
                goal.add(expr == True)
            elif var_type == VarType.ARG:
                arg = get_arg(arg_name)
                if arg is None:
                    continue
                if arg_name == 'padding':
                    if 'VALID' not in arg.arg_prop.enum:
                        goal.add(var.n != z3.BitVecVal(1, BIT_WIDTH))
                    if 'SAME' not in arg.arg_prop.enum:
                        goal.add(var.n != z3.BitVecVal(2, BIT_WIDTH))
                    if 'EXPLICIT' not in arg.arg_prop.enum:
                        goal.add(var.n != z3.BitVecVal(3, BIT_WIDTH))
                elif arg.type == ArgType.BOOL:
                    expr = z3.Or(var.n == z3.BitVecVal(1, BIT_WIDTH), var.n == z3.BitVecVal(0, BIT_WIDTH));
                    goal.add(expr == True)
                elif arg.type == ArgType.INT:
                    if arg_name == 'dim' or arg_name == 'dimension':
                        goal.add(var.n <= z3.BitVecVal(6, BIT_WIDTH))
                        goal.add(var.n >= z3.BitVecVal(0, BIT_WIDTH))
                    else:
                        if prob_bool(0.5):
                            goal.add(var.n < z3.BitVecVal(2 ** 16, BIT_WIDTH))
                            goal.add(var.n > z3.BitVecVal(-2 ** 16, BIT_WIDTH))
                        else:
                            goal.add(var.n < z3.BitVecVal(1024, BIT_WIDTH))
                            goal.add(var.n > z3.BitVecVal(-1024, BIT_WIDTH))
            elif var_type == VarType.TENSOR_DIMSIZE:
                # Make sure the dim is decided in sample process.
                # If no constraints for this arg's dim, then add base constraints on it.
                has_dim = False
                for var in all_vars:
                    n, t = VarType.get_var_type(var)
                    if t == VarType.TENSOR_DIM and n == arg_name:
                        has_dim = True
                        break
                if not has_dim:
                    dim_name = arg_identifier(arg_name) + '_DIM'
                    dim_expr = z3.BitVec(dim_name, BIT_WIDTH)
                    goal.add(dim_expr >= z3.BitVecVal(0, BIT_WIDTH))
                    if prob_bool(0.2):
                        goal.add(dim_expr > z3.BitVecVal(2**16, BIT_WIDTH))
                    else:
                        goal.add(dim_expr <= z3.BitVecVal(6, BIT_WIDTH))
            elif var_type == VarType.LIST_VAL:
                has_len = False
                for var in all_vars:
                    n, t = VarType.get_var_type(var)
                    if t == VarType.LIST_LEN and n == arg_name:
                        has_len = True
                        break
                if not has_len:
                    len_name = arg_identifier(arg_name) + '_LEN'
                    len_expr = z3.BitVec(len_name, BIT_WIDTH)
                    goal.add(len_expr >= z3.BitVecVal(0, BIT_WIDTH))
                    goal.add(len_expr <= z3.BitVecVal(6, BIT_WIDTH))
            elif var_type == VarType.TENSOR_NUM_ELEMENT:
                goal.add(var.n >= z3.BitVecVal(0, BIT_WIDTH))
                goal.add(var.n < z3.BitVecVal(6, BIT_WIDTH))


        vars = get_vars_in_goal(goal)
        pre_arg_set = {}
        for var in vars:
            arg_name, var_type = VarType.get_var_type(var)
            if var_type == VarType.LIST_ITERATOR_VAL:
                opt = z3.Optimize()
                opt.add(goal)
                max_obj = opt.maximize(var.n)
                min_obj = opt.minimize(var.n)
                opt.set('priority', 'box')
                opt.check()
                min_val = min_obj.value().as_long()
                max_val = max_obj.value().as_long()
                if min_val > -128:
                    pre_arg_set[arg_identifier(arg_name) + '_LIST_MIN'] = min_val
                    min_expr = z3.BitVec(arg_identifier(arg_name) + '_LIST_MIN', BIT_WIDTH)
                    goal.add(min_expr == z3.BitVecVal(min_val, BIT_WIDTH))
                if max_val < 128:
                    pre_arg_set[arg_identifier(arg_name) + '_LIST_MAX'] = max_val
                    max_expr = z3.BitVec(arg_identifier(arg_name) + '_LIST_MAX', BIT_WIDTH)
                    goal.add(max_expr == z3.BitVecVal(max_val, BIT_WIDTH))
            elif var_type == VarType.TENSOR_NUM_ELEMENT:
                # Only check if this tensor can have zero element
                opt = z3.Optimize()
                opt.add(goal)
                min_obj = opt.minimize(var.n)
                opt.set('priority', 'box')
                opt.check()
                # print('{} min elements: {}'.format(arg_name, min_obj.value()))
                # input()
                if min_obj.value().as_long() > 0:
                    min_dim = min_obj.value().as_long()
                    pre_arg_set[arg_identifier(arg_name) + '_MINDIM'] = min_dim
                    min_expr = z3.BitVec(arg_identifier(arg_name) + '_MINDIM', BIT_WIDTH)
                    goal.add(min_expr == z3.BitVecVal(min_dim, BIT_WIDTH))

        def set_min_dim(goal, min_dim):
            for select_expr in get_select_exprs_in_goal(goal):
                vars = get_vars(select_expr.n)
                arg_name, var_type = VarType.get_var_type(vars[0])
                # For ksize and strides, their minimal value is zero.
                cur_min = min_dim
                if arg_name == 'ksize' or arg_name == 'strides':
                    cur_min = 1
                if var_type == VarType.TENSOR_DIMSIZE:
                    res_expr = z3.BitVec(get_unique_extend_expr_name(), BIT_WIDTH)
                    goal.add(res_expr == select_expr.n)
                    # If has global minimal value
                    min_identifier = arg_identifier(arg_name) + '_MINDIM'
                    if(min_identifier in pre_arg_set.keys()):
                        goal.add(res_expr >= z3.BitVecVal(pre_arg_set[min_identifier], BIT_WIDTH))
                    else:
                        goal.add(res_expr >= z3.BitVecVal(cur_min, BIT_WIDTH))
                    goal.add(res_expr <= z3.BitVecVal(16, BIT_WIDTH))
            return goal

        def set_list_min(goal, list_min):
            for select_expr in get_select_exprs_in_goal(goal):
                vars = get_vars(select_expr.n)
                arg_name, var_type = VarType.get_var_type(vars[0])
                # For ksize and strides, their minimal value is zero.
                cur_min = list_min
                if arg_name == 'ksize' or arg_name == 'strides':
                    cur_min = 1

                if var_type == VarType.LIST_VAL:
                    res_expr = z3.BitVec(get_unique_extend_expr_name(), BIT_WIDTH)
                    goal.add(res_expr == select_expr.n)

                    max_identifier = arg_identifier(arg_name) + '_LIST_MAX'
                    if(max_identifier in pre_arg_set.keys()):
                        goal.add(res_expr <= z3.BitVecVal(pre_arg_set[max_identifier], BIT_WIDTH))
                    else:
                        goal.add(res_expr <= z3.BitVecVal(128, BIT_WIDTH))

                    min_identifier = arg_identifier(arg_name) + '_LIST_MIN'
                    if(min_identifier in pre_arg_set.keys()):
                        goal.add(res_expr >= z3.BitVecVal(pre_arg_set[min_identifier], BIT_WIDTH))
                    else:
                        goal.add(res_expr >= z3.BitVecVal(cur_min, BIT_WIDTH))


            vars = get_vars_in_goal(goal)
            var_names = [str(var) for var in vars]
            for arg in args:
                cur_min = list_min
                if arg.name == 'ksize' or arg.name == 'strides':
                    cur_min = 1
                if arg.type == ArgType.LIST:
                    max_identifier = arg_identifier(arg.name) + '_LIST_MAX'
                    if max_identifier not in var_names:
                        max_expr = z3.BitVec(max_identifier, BIT_WIDTH)
                        goal.add(max_expr == z3.BitVecVal(128, BIT_WIDTH))

                    min_identifier = arg_identifier(arg.name) + '_LIST_MIN'
                    if min_identifier not in var_names:
                        min_expr = z3.BitVec(min_identifier, BIT_WIDTH)
                        goal.add(min_expr == z3.BitVecVal(cur_min, BIT_WIDTH))
            return goal

        old_goal = z3.Goal()
        for g in goal:
            old_goal.add(g)
        res = []
        goal = set_min_dim(old_goal, 0)
        goal = set_list_min(goal, 0)
        s = z3.Solver()
        s.add(goal)
        res.append(s.to_smt2())
        goal = set_min_dim(old_goal, 1)
        goal = set_list_min(goal, 0)
        s = z3.Solver()
        s.add(goal)
        res.append(s.to_smt2())
        goal = set_min_dim(old_goal, 1)
        goal = set_list_min(goal, 1)
        s = z3.Solver()
        s.add(goal)
        res.append(s.to_smt2())
        goal = set_min_dim(old_goal, 1)
        goal = set_list_min(goal, 1)
        s = z3.Solver()
        s.add(goal)
        res.append(s.to_smt2())

        ret_dict[res_id] = res

    @staticmethod
    def check(idx, goal, ret_dict):
        solver = z3.Solver()
        solver.add(goal)
        res = solver.check()
        ret_dict[idx] = res

    @staticmethod
    def sample_z3(res_id, goal, num, ret_dict, args):
        def get_arg(arg_name):
            for arg in args:
                if arg.name == arg_name:
                    return arg
            return None
        terms = []
        s = z3.Solver()
        s.set("model.compact", False)
        s.add(goal)
        vars = get_vars_in_goal(goal)
        for var in vars:
            arg_name, var_type = VarType.get_var_type(str(var))
            arg = get_arg(arg_name)
            if arg is not None:
                terms.append(var.n)
        a = all_smt(s, terms)
        cnt = 0
        samples = []
        sample_cnt = []
        while(cnt < num):
            arg_set = {}
            m = next(a)
            cnt += 1
            if m is None:
                break
            for var in vars:
                arg_name, var_type = VarType.get_var_type(str(var))
                arg = get_arg(arg_name)
                if arg is None:
                    continue
                if z3.is_array(var.n):
                    l = []
                    v = m[var.n]
                    v = v.children()
                    while(len(v) == 3):
                        idx = v[1].as_long()
                        val = v[2].as_long()
                        if idx > 20:
                            v = v[0].children()
                            continue
                        while idx >= len(l):
                            l.append(None)
                        l[idx] = val
                        v = v[0].children()
                    arg_set[str(var)] = l
                else:
                    arg_set[str(var)] = m.evaluate(var.n).as_long()
            samples.append(arg_set)
            sample_cnt.append(1)
        ret_dict[res_id] = (samples, sample_cnt)

    @staticmethod
    def sample_SMTSampler(res_id, goal, num, ret_dict, sampler):
        solver = z3.Solver()
        solver.add(goal)
        smt_file = os.path.join('/tmp', '{}.z3'.format(res_id))
        with open(smt_file, 'w') as f:
            f.write(solver.to_smt2())

        # print(solver.check())
        # print(solver)
        cmd = '{} -n {} --smtbv {}'.format(sampler, num + 1, smt_file)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL)
        try:
            max_time = 240
            if num + 30.0 < max_time:
                max_time = num + 30.0
            p.wait(max_time)
        except subprocess.TimeoutExpired:
            p.kill()

        sample_file = os.path.join('/tmp', smt_file + '.samples')
        samples = []
        sample_cnt = []
        if not os.path.exists(sample_file):
            os.remove(smt_file)
            ret_dict[res_id] = ([], [])
            return

        with open(sample_file, 'rb') as f:
            sample_lines = f.readlines()
            # Sometimes the sample amount is much greater than the amount we want.
            if len(sample_lines) > num:
                sample_lines = sample_lines[:-1]
            #     sample_lines = np.random.choice(sample_lines, num, replace=False)
            for line in sample_lines:
                arg_set = {}
                line = line.strip()
                line = line[3:]
                line = line.split(b'\x00')

                index = 0
                vars = get_vars_in_goal(goal)
                for i in range(len(vars)):
                    if line[index].startswith(b']'):
                        line[index] = line[index][1:]
                    if line[index].startswith(b'['):
                        length = int(line[index][1:], base=16)
                        index += 2
                        dim_size = []
                        for _ in range(length):
                            idx = int(line[index], base=16)
                            index += 1
                            val = int(line[index], base=16)
                            index += 1
                            if idx > 16:
                                continue
                            # Sometimes sampler would generate non-mentioned list values.
                            # So delete them.
                            if val > 128 or val < 0:
                                continue

                            while len(dim_size) < idx:
                                dim_size.append(None)
                            if idx == len(dim_size):
                                dim_size.append(val)
                            else:
                                dim_size[idx] = val
                        arg_set[str(vars[i])] = dim_size
                    else:
                        arg_set[str(vars[i])] = int(line[index], base=16)
                        index += 1
                exist = False
                for i in range(len(samples)):
                    if samples[i] == arg_set:
                        sample_cnt[i] += 1
                        exist = True
                        break
                if not exist:
                    samples.append(arg_set)
                    sample_cnt.append(1)
                # Stop if we get enough samples
                if len(samples) > num:
                    for i in range(len(sample_cnt)):
                        sample_cnt[i] = 1
                    break
        ret_dict[res_id] = (samples, sample_cnt)

        # Clean
        os.remove(smt_file)
        os.remove(sample_file)
        return ret_dict



class FuzzManager:
    def __init__(self, op_name, api_name, constraint_files, test_option: TestOption):
        self.op_name = op_name
        self.api_name = api_name
        self.z3files = constraint_files

        self.constraints = []

        self.test_option = test_option
        # Prepare constraints.
        print('Test', self.op_name, ', api:', self.api_name)
        self.read_op_json()

        if self.test_option.use_cons:
            self.clear_z3file()
            self.constraint_extension()

            # Random uniform sample.
            self.samples = []       # Sample result list
            self.test_times = []    # How many times should the sample be tested.
            self.sample_goals = []
            if len(self.constraints) == 0:
                # if len(self.args) > 2:
                #     return
                self.samples = [None]
                self.test_times = [self.test_option.test_round]
            else:
                self.sample()

            ratio = self.test_option.test_round / sum(self.test_times)
            for i in range(len(self.test_times)):
                self.test_times[i] = int(self.test_times[i] * ratio) + 1
        else:
            self.samples = [None]
            self.test_times = [self.test_option.test_round]


        # Fuzz
        mode = self.test_option.mode
        st_t = time.time()
        if mode == 'all' or mode == 'gpu':
            print('\n================== Fuzz {} in {} mode, {} samples ====================\n'.format(self.op_name, 'gpu', len(self.samples)))
            runner = Runner(api_name=self.api_name, mode='gpu', test_option=self.test_option)
            for i in range(len(self.samples)):
                self.apply_sample(self.samples[i])
                manager = APIManager(self.api_name, self.args, self.test_option)
                runner.run(manager, self.test_times[i])
                if runner.terminated:
                    break
            runner.terminate()
        if mode == 'all' or mode == 'cpu_ori':
            print('\n================== Fuzz {} in CPU mode, oneDNN:0 ====================\n'.format(self.op_name))
            runner = Runner(api_name=self.api_name, mode='cpu_ori', test_option=self.test_option)
            for i in range(len(self.samples)):
                self.apply_sample(self.samples[i])
                manager = APIManager(self.api_name, self.args, self.test_option)
                runner.run(manager, self.test_times[i])
                if runner.terminated:
                    break
            runner.terminate()
        if mode == 'cov':
            print('\n================== Fuzz {} in CPU mode, oneDNN:0 ====================\n'.format(self.op_name))
            print('Call runner')
            runner = Runner(api_name=self.api_name, mode='cov', test_option=self.test_option)
            for i in range(len(self.samples)):
                self.apply_sample(self.samples[i])
                manager = APIManager(self.api_name, self.args, self.test_option)
                # input()
                runner.run(manager, self.test_times[i])
                if runner.terminated:
                    break
            runner.terminate()
        if mode == 'all' or mode == 'cpu_onednn':
            if self.test_option.fw == 'torch':
                return
            print('\n================== Fuzz {} in CPU mode, oneDNN:1 ====================\n'.format(self.op_name))
            runner = Runner(api_name=self.api_name, mode='cpu_onednn', test_option=self.test_option)
            for i in range(len(self.samples)):
                self.apply_sample(self.samples[i])
                manager = APIManager(self.api_name, self.args, self.test_option)
                runner.run(manager, self.test_times[i])
                if runner.terminated:
                    break
            runner.terminate()
        print()
        print()
        print()
        print(time.time()-st_t)


    def sample(self):
        # test at least 2 times for each sample
        test_round = self.test_option.test_round
        if len(self.constraints) > test_round / 2:
            self.constraints = random.sample(self.constraints, k=int(test_round/2))

        sample_per_cons = int(test_round / len(self.constraints) / 2)
        if sample_per_cons < 2:
            sample_per_cons = 2
        print('Sample {} constraints'.format(len(self.constraints)))
        sampler = MultiProcessHandler(self.constraints, sample_per_cons, task='sample', test_option=self.test_option, args=self.args)
        sampler.execute()

        total_sample_num = 0
        unique_sample_num = 0
        for idx in range(len(self.constraints)):
            if sampler.res[idx] is None:
                continue
            res = sampler.res[idx][0]
            cnt = sampler.res[idx][1]
            total_sample_num += len(res)
            for i in range(len(res)):
                # dedup
                duplicate = False
                for j in range(len(self.samples)):
                    if self.samples[j] == res[i]:
                        self.test_times[j] += cnt[i]
                        duplicate = True
                if not duplicate:
                    unique_sample_num += 1
                    self.samples.append(res[i])
                    self.test_times.append(cnt[i])
                    self.sample_goals.append(self.constraints[idx])
        print('Total {} samples, {} unique in all.'.format(total_sample_num, unique_sample_num))

        if len(self.samples) == 0:
            return

        # If the amount of constraints is much bigger than the
        # amount of samples we've got, sample more samples from
        # fewer constraints.
        # This may happen when there are a lot of similar constraints.
        if len(self.constraints) / len(self.samples) > 5:
            print('Too many same samples, Resample from fewer constraints...')
            # Reset test time of previous sample
            self.test_times = [1 for _ in range(len(self.test_times))]

            cons_number = int(len(self.constraints) / 20)
            if cons_number < 5:
                cons_number = 5
            choosed_cons = random.sample(self.constraints, k=cons_number)

            sampler = MultiProcessHandler(choosed_cons, int(test_round / cons_number)*2, task='sample', test_option=self.test_option, args=self.args)
            sampler.execute()

            for r in sampler.res:
                if r is None:
                    continue
                res = r[0]
                cnt = r[1]
                for i in range(len(res)):
                    # dedup
                    duplicate = False
                    for j in range(len(self.samples)):
                        if self.samples[j] == res[i]:
                            self.test_times[j] += cnt[i]
                            duplicate = True
                    if not duplicate:
                        self.samples.append(res[i])
                        self.test_times.append(cnt[i])
                        self.sample_goals.append(self.constraints[idx])
        print('Total {} unique samples.'.format(len(self.samples)))
        # Shutffle
        t = list(zip(self.test_times, self.samples))
        random.shuffle(t)
        self.test_times[:], self.samples[:] = zip(*t)



    def apply_sample(self, sample):
        # Reset args.
        self.read_op_json()
        if sample is None:
            return

        # Check if some of tensors are actually list or scalar.
        # for var_name in sample.keys():
        #     arg_name, var_type = VarType.get_var_type(var_name)
        #     arg = self.get_arg(arg_name)
        #     if arg is None:
        #         print(arg_name)
        #         assert(0)

        #     if arg.type != ArgType.TENSOR:
        #         continue
        #     if var_type == VarType.LIST_VAL or \
        #         var_type == VarType.LIST_MIN or \
        #             var_type == VarType.LIST_MAX or \
        #                 var_type == VarType.LIST_LEN:
        #         arg.type = ArgType.LIST
        #         arg.arg_prop = arg.arg_prop.to_list()
        #     elif var_type == VarType.ARG:
        #         arg.arg_prop = arg.arg_prop.to_scalar()
        #         arg.type = arg.arg_prop.type

        # Apply constraints to prop
        for var_name in sample.keys():
            arg_name, var_type = VarType.get_var_type(var_name)
            if var_type == VarType.VAR or \
                var_type == VarType.EXTEND_VAR or \
                var_type == VarType.LIST_ITERATOR_VAL or\
                var_type == VarType.TENSOR_NUM_ELEMENT:
                continue
            arg = self.get_arg(arg_name)
            if arg is None:
                continue
            arg.arg_prop.use_default = False
            if arg.type == ArgType.STRING:
                if var_type == VarType.STR_LEN:
                    arg.arg_prop.length = sample[var_name]
                elif var_type == VarType.STR_FMT:
                    enum = int_to_fmt(sample[var_name])
                    for e in enum:
                        if e in arg.arg_prop.enum:
                            arg.arg_prop.value = e
                            break
                    assert(arg.arg_prop.value != None)
                elif var_type == VarType.ARG:
                    if arg_name == 'padding':
                        if sample[var_name] == 1:
                            arg.arg_prop.value = 'VALID'
                        elif sample[var_name] == 2:
                            arg.arg_prop.value = 'SAME'
                        elif sample[var_name] == 3:
                            arg.arg_prop.value = 'EXPLICIT'
                    else:
                        if len(arg.arg_prop.enum) > 0:
                            arg.arg_prop.value = arg.arg_prop.enum[sample[var_name] -1]

                else:
                    continue
            elif arg.type == ArgType.LIST:
                if var_type == VarType.LIST_LEN:
                    # or var_type == VarType.TENSOR_DIM:
                    arg.arg_prop.length = sample[var_name]
                elif var_type == VarType.LIST_VAL:
                    len_var_name = arg_identifier(arg_name) + '_LEN'
                    length = sample[len_var_name]
                    # length = None
                    # if len_var_name in sample.keys():
                    #     length = sample[len_var_name]
                    # else:
                    #     length = sample[arg_identifier(arg_name) + '_DIM']
                    arg.arg_prop.value = sample[var_name]
                    while(len(arg.arg_prop.value) < length):
                        arg.arg_prop.value.append(None)
                    while(len(arg.arg_prop.value) > length):
                        arg.arg_prop.value.pop()
                elif var_type == VarType.LIST_MIN:
                    if type(arg.arg_prop.child_prop) is list:
                        for c in arg.arg_prop.child_prop:
                            c['min'] = sample[var_name]
                    else:
                        arg.arg_prop.child_prop['min'] = sample[var_name]
                elif var_type == VarType.LIST_MAX:
                    if type(arg.arg_prop.child_prop) is list:
                        for c in arg.arg_prop.child_prop:
                            c['max'] = sample[var_name]
                    else:
                        arg.arg_prop.child_prop['max'] = sample[var_name]
                elif var_type == VarType.TENSOR_DIMSIZE:
                    if len(sample[var_name]) > 0:
                        arg.arg_prop.length = sample[var_name][0]
                elif var_type == VarType.TENSOR_DIM:
                    continue
                else:
                    print(var_name, arg_name)
                    assert(0)
            elif arg.type == ArgType.TENSOR:
                if var_type == VarType.TENSOR_DIM:
                    # or var_type == VarType.LIST_LEN:
                    if sample[var_name] > 6:
                        sample[var_name] = 6
                    arg.arg_prop.ndim = sample[var_name]
                elif var_type == VarType.TENSOR_DIMSIZE:
                    dim_var_name = arg_identifier(arg_name) + '_DIM'
                    ndim = sample[dim_var_name]
                    if ndim > 6:
                        ndim = 6
                    # There might be a bug in Constraint Extraction which mistakes DIMSIZE as int
                    if type(sample[var_name]) != list:
                        continue
                    for i in range(len(sample[var_name])):
                        if sample[var_name][i] is not None and sample[var_name][i] > 16:
                            sample[var_name][i] = None
                    arg.arg_prop.shape = sample[var_name]
                    while(len(arg.arg_prop.shape) < ndim):
                        arg.arg_prop.shape.append(None)
                    while(len(arg.arg_prop.shape) > ndim):
                        arg.arg_prop.shape.pop()
                elif var_type == VarType.TENSOR_MIN_DIM:
                    arg.arg_prop.min_ndim = sample[var_name]
                else:
                    print(arg_name, var_name)
                    assert(0)

            elif arg.type == ArgType.INT:
                if var_type == VarType.ARG:
                    arg.arg_prop.value = sample[var_name]
                    if arg.arg_prop.value > 2 ** 32:
                        arg.arg_prop.value -= 2 ** 64
                else:
                    continue
            elif arg.type == ArgType.FLOAT:
                if var_type == VarType.ARG:
                    arg.arg_prop.value = sample[var_name]
                else:
                    continue
            elif arg.type == ArgType.BOOL:
                if var_type == VarType.ARG:
                    arg.arg_prop.value = sample[var_name]
                else:
                    continue
            elif arg.type == ArgType.TYPE:
                if var_type == VarType.ARG:
                    arg.arg_prop.value = sample[var_name]
                else:
                    continue
            else:
                print(arg.type)
                assert(0)

        # If a tensor is 1-dim, set its elements number
        for var_name in sample.keys():
            arg_name, var_type = VarType.get_var_type(var_name)
            if var_type == VarType.TENSOR_NUM_ELEMENT:
                arg = self.get_arg(arg_name)
                if arg is None:
                    continue
                if arg.arg_prop.ndim == 1:
                    arg.arg_prop.shape = [sample[var_name]]
                elif sample[var_name] == 1:
                    arg.arg_prop.shape = [1]

    def get_arg(self, arg_name):
        for arg in self.args:
            if arg.name == arg_name:
                return arg
        return None


    # Add base constraints for vars in constraints
    def constraint_extension(self):

        print('Extend constraints...')
        extended_constraints = []
        extender = MultiProcessHandler(self.constraints, task='extend', args=self.args, test_option=self.test_option)
        extender.execute()
        for i in range(len(self.constraints)):
            for goal in extender.res[i]:
                extended_constraints.append(z3.parse_smt2_string(goal))

        print('Check the satisfiability of {} extended constraints...'.format(len(extended_constraints)))
        checker = MultiProcessHandler(extended_constraints, task='check', test_option=self.test_option)
        checker.execute()
        self.constraints.clear()
        for i in range(len(extended_constraints)):
            if checker.res[i] == z3.sat:
                self.constraints.append(extended_constraints[i])
            # else:
            #     for c in extended_constraints[i]:
            #         print(c)
            #     input()


        print(len(self.constraints), 'constraints left after extension')

    def rm_uninteresting_exprs(self, goal):
        new_goal = z3.Goal()
        interesting_expr = []
        for idx in range(goal.size()):
            expr = goal.get(idx)
            vars = get_vars(expr)
            interesting = False
            for var in vars:
                if VarType.get_var_type(var)[1] != VarType.VAR:
                    interesting = True
                    new_goal.add(expr)
                    break
            if interesting:
                interesting_expr.append(True)
            else:
                interesting_expr.append(False)

        interesting_vars = set()
        for expr in new_goal:
            vars = get_vars(expr)
            for var in vars:
                interesting_vars.add(var)

        changed = True
        while changed:
            changed = False
            # traverse uninteresting exprs, if it contains interesting vars,
            # mark it as interesting
            for idx in range(goal.size()):
                if interesting_expr[idx]:
                    continue
                expr = goal.get(idx)
                vars = get_vars(expr)
                interesting = False
                for var in vars:
                    if var in interesting_vars:
                        interesting = True
                        break
                if interesting:
                    changed = True
                    interesting_expr[idx] = True
                    new_goal.add(expr)
                    for var in vars:
                        interesting_vars.add(var)
        return new_goal




    def clear_z3file(self):
        print('Clear rebundant constraint files...')
        list_tensors = set()
        scalar_tensors = set()
        if self.test_option.fw == 'tf':
            compute_files = []
            constructor_files = []
            for z3file in self.z3files:
                if z3file.find('compute') != -1:
                    compute_files.append(z3file)
                else:
                    constructor_files.append(z3file)
            if len(compute_files) > 100:
                compute_files = random.sample(compute_files, k=100)
            if len(constructor_files) > 100:
                constructor_files = random.sample(constructor_files, k=100)
            self.z3files = compute_files + constructor_files
        elif self.test_option.fw == 'torch':
            if len(self.z3files) > int(self.test_option.test_round / 2):
                self.z3files = random.sample(self.z3files, k=int(self.test_option.test_round / 2))

        removed_files = set()
        if self.test_option.fw == 'tf':
            unique_compute_md5 = set()
            unique_constructor_md5 = set()
            for z3file in self.z3files:
                f = open(z3file,'rb')
                md5 = hashlib.md5(f.read()).hexdigest()
                f.close()

                # remove duplicate files
                if z3file.find('compute') != -1:
                    if md5 in unique_compute_md5:
                        os.remove(z3file)
                        removed_files.add(z3file)
                    else:
                        unique_compute_md5.add(md5)
                elif z3file.find('constructor') != -1:
                    if md5 in unique_constructor_md5:
                        os.remove(z3file)
                        removed_files.add(z3file)
                    else:
                        unique_constructor_md5.add(md5)
        elif self.test_option.fw == 'torch':
            unique_md5 = set()
            for z3file in self.z3files:
                f = open(z3file,'rb')
                md5 = hashlib.md5(f.read()).hexdigest()
                f.close()

                if md5 in unique_md5:
                    os.remove(z3file)
                    removed_files.add(z3file)
        for f in removed_files:
            if f in self.z3files:
                self.z3files.remove(f)


        compute_constraints = []
        constructor_constraints = []
        for z3file in self.z3files:
            if not os.path.exists(z3file):
                continue
            f = open(z3file,'r')
            content = f.read()
            f.close()
            # Replace input_x as its real name.
            for k in self.input_arg_dict.keys():
                content = content.replace(k, self.input_arg_dict[k])

            # Some lists are marked as tensor, convert its defination and
            # constraints to list.
            for arg in self.args:
                arg_id = arg_identifier(arg.name)
                if arg.type == ArgType.TENSOR:
                    # Args end with shape are definately list-type.
                    if arg.name.endswith('_shape') or arg.name == 'shape':
                        content = content.replace(arg_id + '_DIM', get_unique_extend_expr_name())
                        content = content.replace(arg_id + '_NUMELEMENT', arg_id + '_LEN')
                        list_tensors.add(arg.name)
                    # Args used as an argument in vector-funcs are list-type.
                    elif content.find(arg_id + '_LEN') != -1 or \
                        content.find(arg_id + '_VAL') != -1 or \
                            content.find(arg_id + '_ITERATOR_VAL') != -1:
                        # Make sure there is no confusion on its type
                        if content.find(arg_id + '_DIM') != -1 or \
                            content.find(arg_id + '_DIMSIZE') != -1:
                            assert(0)
                        list_tensors.add(arg.name)
                    elif content.find(arg_id + '_SCALAR') != -1:
                        content = content.replace(arg_id + '_DIMSIZE', get_unique_extend_expr_name())
                        content = content.replace(arg_id + '_DIM', get_unique_extend_expr_name())
                        content = content.replace(arg_id + '_NUMELEMENT', get_unique_extend_expr_name())
                        content = content.replace(arg_id + '_SCALAR', arg_id)
                        scalar_tensors.add(arg.name)
                    elif content.find(arg_id + ' ') != -1:
                        scalar_tensors.add(arg.name)
                        content = content.replace(arg_id + '_DIMSIZE', get_unique_extend_expr_name())
                        content = content.replace(arg_id + '_DIM', get_unique_extend_expr_name())
                        content = content.replace(arg_id + '_NUMELEMENT', get_unique_extend_expr_name())
                        content = content.replace(arg_id + '_SCALAR', arg_id)
                elif arg.type == ArgType.LIST:
                    content = content.replace(arg_id + '_DIM', get_unique_extend_expr_name())
                    content = content.replace(arg_id + '_NUMELEMENT', arg_id + '_LEN')
                elif arg.type == ArgType.INT:
                    content = content.replace(arg_id + '_SCALAR', arg_id)
                    content = content.replace(arg_id + '_DIMSIZE', get_unique_extend_expr_name())
                    content = content.replace(arg_id + '_DIM', get_unique_extend_expr_name())
                    content = content.replace(arg_id + '_SHAPE', get_unique_extend_expr_name())

            # Remove duplicate decalre

            content = content.split('\n')
            declared_vars = set()
            for i in range(len(content)):
                res = re.match(r'\(declare-fun ([a-zA-Z0-9_]+) \(\) .*', content[i])
                if res is None:
                    continue
                if res.group(1) not in declared_vars:
                    declared_vars.add(res.group(1))
                else:
                    content[i] = ''

            f = open(z3file,'w')
            for line in content:
                f.write(line + '\n')
            f.close()

            constraint = z3.parse_smt2_file(z3file, sorts={}, decls={})

            # remove unsolvable constraints
            goal = z3.Goal()
            for c in constraint:
                goal.add(c)

            if len(goal) == 0:
                continue

            if self.test_option.fw == 'tf':
                if z3file.find('compute') != -1:
                    compute_constraints.append(goal)
                elif z3file.find('constructor') != -1:
                    constructor_constraints.append(goal)
            elif self.test_option.fw == 'torch':
                compute_constraints.append(goal)

        # Rewrite tensors who are actually list-type
        for arg_name in list_tensors:
            arg = self.get_arg(arg_name)
            arg.arg_prop = arg.arg_prop.to_list()
            arg.type = ArgType.LIST
        # Rewrite tensors who are actually scalar-type
        for arg_name in scalar_tensors:
            arg = self.get_arg(arg_name)
            arg.arg_prop = arg.arg_prop.to_scalar()
            arg.type = arg.arg_prop.type

        print('Dedup finished: {} compute constraints, {} constructor constraints.'.format(len(compute_constraints), len(constructor_constraints)))

        self.rewrite_json()
        self.read_op_json()

        # Combine constraints from different compute and constructor
        combined_goals = []
        for comp_cons in compute_constraints:
            for con_cons in constructor_constraints:
                g = z3.Goal()
                for sub_goal in comp_cons:
                    g.add(sub_goal)
                for sub_goal in con_cons:
                    g.add(sub_goal)
                combined_goals.append(g)
        if len(constructor_constraints) == 0:
            for comp_cons in compute_constraints:
                goal = z3.Goal()
                for sub_goal in comp_cons:
                    goal.add(sub_goal)
                combined_goals.append(goal)
        if len(compute_constraints) == 0:
            for con_cons in constructor_constraints:
                goal = z3.Goal()
                for sub_goal in con_cons:
                    goal.add(sub_goal)
                combined_goals.append(goal)
        # Remove rebundant constraints
        if len(combined_goals) > self.test_option.test_round:
            combined_goals = random.sample(combined_goals, k=self.test_option.test_round)
        
        # Remove uninteresting exprs
        print('Remove uninteresting exprs in each goal...')
        for i in range(len(combined_goals)):
            print('{}/{}'.format(i, len(combined_goals)), end='\r')
            combined_goals[i] = self.rm_uninteresting_exprs(combined_goals[i])
        # Only in TensorFlow, a goal is combined with two sets of satisfiable constraints and might be unsatisfiable
        if self.test_option.fw == 'tf':
            print('Check satisfiability for each of {} combinations...'.format(len(combined_goals)), flush=True)

            checker = MultiProcessHandler(combined_goals, task='check', test_option=self.test_option)
            checker.execute()
            for i in range(len(combined_goals)):
                if checker.res[i] == z3.sat:
                    self.constraints.append(combined_goals[i])
        else:
            self.constraints = combined_goals

        print('Unique constraint files of {}:'.format(self.op_name), len(compute_constraints) + len(constructor_constraints), '/', len(self.z3files))
        print(len(self.constraints), 'satisfied combinations for {} compute constraints and {} constructor constraints.' \
            .format(len(compute_constraints), len(constructor_constraints)))

    def read_op_json(self):
        json_name = self.api_name
        if self.test_option.fw == 'torch':
            json_name = self.op_name
        elif self.test_option.fw == 'tf':
            json_name = self.api_name.split('.')[-1]
        auto_op_file = os.path.join(self.test_option.api_dir, '{}.json'.format(json_name))
        # print('Reading auto generated json files:', auto_op_file)
        if not os.path.exists(auto_op_file):
            print('Auto generated op info not found:', auto_op_file)
            exit(-1)
        with open(auto_op_file, 'r') as f:
            self.arg_dict = json.load(f)
        self.input_arg_dict = {}
        self.args = []
        for arg_name in self.arg_dict.keys():
            self.args.append(Argument(arg_name, self.arg_dict[arg_name]))
            if self.arg_dict[arg_name]['input_order'] != -1:
                self.input_arg_dict['input_{}'.format(self.arg_dict[arg_name]['input_order'])] = arg_name

    def rewrite_json(self):
        json_name = self.api_name.replace('tf.raw_ops.', '')
        if self.test_option.fw == 'torch':
            json_name = self.op_name
        auto_op_file = os.path.join(self.test_option.api_dir, '{}.json'.format(json_name))
        d = {}
        for arg in self.args:
            d[arg.name] = arg.arg_prop.to_dict()
        with open(auto_op_file, 'w') as f:
            json.dump(d, f, indent='\t')
        print('Rewrite', auto_op_file)

def fuzz_op(op_name, api_name, test_option: TestOption):
    op_dir = os.path.join(test_option.constraint_dir, op_name)
    z3_files = []
    if os.path.exists(op_dir):
        for file_name in os.listdir(op_dir):
            if file_name.endswith('.z3'):
                z3_files.append(os.path.join(op_dir, file_name))
    f = FuzzManager(op_name=op_name, api_name=api_name, constraint_files=z3_files, test_option=test_option)
    return

    try:
        f = FuzzManager(op_name=op_name, api_name=api_name, constraint_files=z3_files, test_option=test_option)
    except(KeyboardInterrupt) as e:
        traceback.print_exc()
        exit(0)
    except:
        print('Failed!!')
        with open(os.path.join(test_option.output_path, 'res.csv'), 'a') as f:
            f.write('{} {} {} Failed\n'.format(op_name, api_name, test_option.mode))


def get_op_name(api_name, api2op_csv):
    with open(api2op_csv, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            api = line.split(' ')[0]
            if api == api_name:
                return line.split(' ')[1]
    return None


if __name__ == '__main__':

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, '',
                                   ['framework=', 'filter=', 'mode=',
                                        'test_round=', 'round_timeout=',
                                        'use_cons=', 'api_timeout=',
                                        'target_api=', 'save_non_crash=',
                                        'total_round=', 'get_cov=', 'work_path=',
                                         "save_bitmap=", 'record_cov='
                                        ])
    except:
        print('Error: invalid options.')
    test_opt = TestOption()


    for opt, arg in opts:
        if opt == '--total_round':
            test_opt.total_round = int(arg)
        if opt == '--work_path':
            test_opt.work_path = arg
        if opt == '--framework':
            test_opt.fw = arg
        if opt == '--filter':
            test_opt.filter = arg
        if opt == '--mode':
            test_opt.mode = arg
        if opt == '--test_round':
            test_opt.test_round = int(arg)
        if opt == '--round_timeout':
            test_opt.round_timeout = int(arg)
        if opt == '--get_cov':
            if arg == 'true':
                test_opt.get_cov = True
                print('Get Coverage.')
        if opt == '--record_cov':
            if arg == 'true':
                test_opt.record_cov = True
                print('Record Coverage.')
        if opt == '--use_cons':
            if arg == 'true':
                test_opt.use_cons = True
            else:
                test_opt.use_cons = False
                print('Not using cons.')
        if opt == '--api_timeout':
            test_opt.api = arg
        if opt == '--target_api':
            test_opt.target_api = arg
        if opt == '--save_bitmap':
            if arg == 'true':
                test_opt.save_bitmap = True
        if opt == '--save_non_crash':
            if arg == 'true':
                test_opt.save_non_crash = True
                print('Save non-crash test cases.')
            else:
                test_opt.save_non_crash = False

    test_opt.initialize()

    api_list = []
    if test_opt.fw == 'tf':
        api_list = [ 'tf.raw_ops.' + fname.replace('.json', '') for fname in os.listdir(test_opt.api_dir) ]
    elif test_opt.fw == 'torch':
        with open(test_opt.api2op_csv, 'r') as f:
            for line in f.readlines():
                line = line.split(' ')[0]
                if line.startswith('torch'):
                    api_list.append(line)
    api_list.sort()

    op_list = os.listdir(test_opt.constraint_dir)
    op_list.sort()

    test_targets = []

    filter = test_opt.filter
    if filter == 'all':
        for api_name in api_list:
            op_name = get_op_name(api_name, test_opt.api2op_csv)
            if op_name is None:
                continue
            test_targets.append([op_name, api_name])
    elif filter == 'new':
        tested_list = os.listdir(test_opt.output_path)
        for api_name in api_list:
            if api_name in tested_list:
                continue
            op_name = get_op_name(api_name, test_opt.api2op_csv)
            if op_name is None:
                continue
            test_targets.append([op_name, api_name])
    elif filter == 'exist':
        for api_name in api_list:
            op_name = get_op_name(api_name, test_opt.api2op_csv)
            if op_name not in op_list:
                continue
            test_targets.append([op_name, api_name])
    elif filter == 'after':
        assert(test_opt.target_api != None)
        reached = False
        for api_name in api_list:
            op_name = get_op_name(api_name, test_opt.api2op_csv)
            if op_name is None:
                continue
            if api_name == test_opt.target_api:
                reached = True
                continue
            if not reached:
                continue
            test_targets.append([op_name, api_name])
    elif filter == 'list':
        with open('op_list.txt', 'r') as f:
            tested_list = os.listdir(test_opt.output_path)
            for line in f.readlines():
                api_name = line.strip()
                # if api_name in tested_list:
                #     continue
                op_name = get_op_name(api_name, test_opt.api2op_csv)
                if op_name is None:
                    continue
                if [op_name, api_name] not in test_targets:
                    test_targets.append([op_name, api_name])
    elif filter == None:
        op_name = get_op_name(test_opt.target_api, test_opt.api2op_csv)
        if op_name is None:
            print('API not found:', test_opt.target_api)
            exit(0)
        test_targets.append([op_name, test_opt.target_api])
    
    if test_opt.fw == 'torch':
        defined_ops = [ f.replace('.json', '')  for f in os.listdir(test_opt.api_dir) ]
        
        temp = []
        for target in test_targets:
            if target[0] in defined_ops or target[0].replace('Op', '') in defined_ops or target[0] + 'Op' in defined_ops:
                temp.append(target)
        test_targets = temp
        

    for i in range(test_opt.total_round):
        test_opt.initialize(round=i)
        start_time = time.time()
        print('{} targets to test.'.format(len(test_targets)))

        tested_list = os.listdir(test_opt.output_path)
        for target in test_targets:
            if target[1] in tested_list:
                continue
            fuzz_op(op_name = target[0], api_name=target[1], test_option=test_opt)
        print('Test finished')
        print(time.time() - start_time, 's in total')
