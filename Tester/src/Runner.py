
from termcolor import colored
import os
import time

from queue import Empty
from multiprocessing import Process, Queue
import subprocess
import json
import z3
from Option import TestOption

import sysv_ipc
import ctypes, ctypes.util

# AFL Shared Memory set
SHM_ENV_VAR   = "__AFL_SHM_ID"
MAP_SIZE = 1 << 23
FORKSRV_FD = 198

class Result:
    def __init__(self, exception=False, except_str=None, err_msg=None):
        self.exception = exception
        self.except_str = except_str
        self.err_msg = err_msg

def exec_code_tf(code_q, res_q, mode, get_cov=False):
    import os
    if mode == 'cpu_onednn':
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    else:
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    if get_cov:
        from ctypes import CDLL
        import sys
        rt_funcs = CDLL('../data/afl-llvm-rt.so')
        rt_funcs.__afl_map_shm()

    import tensorflow as tf
    while(True):
        try:
            code = None
            code = code_q.get(timeout=30)
            local = {'ends_with_exception': False, 'exception_str': None}
            exec(code, {}, local)
            res = Result(exception=local['ends_with_exception'], except_str=local['exception_str'])
            res_q.put(res)
        except Empty as e:
            print('subprocess: EMPTY CODE_QUEUE')
            res = Result(exception=True, except_str='NO_INPUT')
            res_q.put(res)
            return
        except Exception as e:
            print('subprocess: UNKNOWN EXEPTION IN EXEC:' + str(e))
            res = Result(exception=True, except_str=str(e))
            res_q.put(res)

def exec_code_torch(code_q, res_q, mode, get_cov=False):
    import torch
    while(True):
        try:
            code = None
            code = code_q.get(timeout=30)
            local = {'ends_with_exception': False, 'exception_str': None}
            exec(code, {}, local)
            res = Result(exception=local['ends_with_exception'], except_str=local['exception_str'])
            res_q.put(res)
        except Empty as e:
            print('subprocess: EMPTY CODE_QUEUE')
            res = Result(exception=True, except_str='NO_INPUT')
            res_q.put(res)
            return
        except Exception as e:
            print('subprocess: UNKNOWN EXEPTION IN EXEC:' + str(e))
            res = Result(exception=True, except_str=str(e))
            res_q.put(res)

class Runner:
    def __init__(self, api_name,  test_option: TestOption, mode='cpu_ori'):
        self.start_time = time.time()
        self.terminated = False
        self.index = 0
        self.total_times = 0
        self.invalid = 0
        self.crash = 0
        self.timeout = 0
        self.OOM = 0
        self.mode = mode
        self.api_name = api_name
        self.test_option = test_option
        self.non_crash = 0

        self.save_bmp = test_option.save_bitmap
        self.get_cov = test_option.get_cov or self.save_bmp
        self.coverage = 0
        if self.get_cov:
            libc_path = ctypes.util.find_library("c")
            if not libc_path:
                sys.exit("cannot find libc")
            # load libc
            self.libc = ctypes.CDLL(libc_path)
            if not self.libc:
                sys.exit("cannot load libc")
            self.setup_shm()


        self.cpu_gpu_split = '\n# ====================CPU_GPU_DIFF====================\n'

        self.generate_output_path()
        self.runner_p = None
        # self.init_exec_process()

    def generate_output_path(self):
        self.output_dir = os.path.join(self.test_option.output_path, self.api_name)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        for sub_dir in ['crash', 'timeout', 'samples', 'invalid', 'non_crash']:
            outp = os.path.join(self.output_dir, sub_dir)
            if not os.path.exists(outp):
                os.mkdir(outp)

    def save_error_code(self, index, code, err_type):
        outp = os.path.join(self.output_dir, err_type)
        sub_idx = 0
        if err_type == 'crash':
            sub_idx = self.crash
        elif err_type == 'timeout':
            sub_idx = self.timeout
        elif err_type == 'invalid':
            sub_idx = self.invalid
        elif err_type == 'non_crash':
            sub_idx = self.non_crash
        else:
            assert(0)
        out_f = os.path.join(outp, '{}_{}_{}.py'.format(self.mode, index, sub_idx))
        with open(out_f, 'w') as f:
            f.write(code)
        if err_type == 'crash':
            with open(os.path.join(outp, 'crash_summary.csv'), 'a') as f:
                f.write('{}, {}, {}, {}\n'.format(self.api_name, self.mode, out_f, self.runner_p.exitcode))

    def init_exec_process(self):
        if self.runner_p is not None:
            if self.runner_p.is_alive():
                self.runner_p.kill()
        self.code_q = Queue()
        self.result_q = Queue()
        target_func = exec_code_tf 
        if self.test_option.fw == 'torch':
            target_func = exec_code_torch
        self.runner_p = Process(target=target_func, \
            args=(self.code_q, self.result_q, self.mode, self.get_cov))
        self.runner_p.start()
        # wait until package is loaded
        time.sleep(7)

    def run(self, manager, times):
        cur_invalid = 0
        cur_crash = 0
        cur_timeout = 0
        self.index += 1
        with open(os.path.join(self.output_dir, 'samples', '{}.json'.format(self.index)), 'w') as f:
            json.dump(manager.to_dict(), f, indent='\t')

        for t in range(times):
            if time.time() - self.start_time > self.test_option.api_timeout:
                print('Max test time {}s for this API has been reached'.format(self.test_option.api_timeout))
                self.terminate()
                break
            if self.crash >= self.test_option.max_api_error or self.timeout >= self.test_option.max_api_error:
                print('Crash or Timeout limit has been reached.')
                self.terminate()
                break
            elif self.test_option.test_round <= self.total_times:
                break

            if self.total_times % 50 == 0 and self.get_cov:
                self.count_cov()
            print(colored('======Testing {} in mode {} ----Sample Id: {}, Total:{}, Invalid: {}, Crash: {}, Timeout:{}, Coverage:{} ======'
                .format(self.api_name, self.mode, self.index, self.total_times, self.invalid, self.crash, self.timeout, self.coverage), 'blue'),
                    end= '\r' if self.total_times % 50 != 0 else '\n')
            self.total_times += 1


            manager.generate_random_args()
            code = None
            if self.mode == 'cpu_ori' or self.mode == 'cpu_onednn' or self.mode ==  'cov':
                code = manager.get_cpu_code()
            elif self.mode == 'gpu':
                code = manager.get_gpu_code()


            if self.runner_p is None or not self.runner_p.is_alive():
                self.init_exec_process()
            self.code_q.put(code)

            res = None
            timeout = False
            crash = False
            cur_pass = True
            counter = 0
            try:
                while(counter < self.test_option.round_timeout):
                    counter += 1
                    try:
                        res = self.result_q.get(timeout=1)
                        break
                    except Empty as e:
                        if not self.runner_p.is_alive():
                            print("Error: Subprocess is dead.")
                            crash = True
                            break
                if res is None and self.runner_p.is_alive():
                    self.runner_p.kill()
                    timeout = True
                    # Go on after the failed process is completely killed
                    time.sleep(1)
                    while(self.runner_p.is_alive()):
                        time.sleep(0.1)
                        self.runner_p.kill()
                        print('Try to kill TIMEOUT process')

            except Exception as e:
                if self.runner_p.is_alive():
                    self.runner_p.kill()
                    # Go on after the failed process is completely killed
                    time.sleep(1)
                    while(self.runner_p.is_alive()):
                        time.sleep(0.1)
                        self.runner_p.kill()
                        print('Try to kill Exception process')
                print('ERROR:', str(e))
                crash = True
            else:
                if not self.runner_p.is_alive() and self.runner_p.exitcode != 0:
                    crash = True
            if timeout:
                if cur_timeout < self.test_option.max_sample_error:
                    self.save_error_code(self.index, code, 'timeout')
                else:
                    break
                self.timeout += 1
                cur_timeout += 1
            elif crash:
                if cur_crash < self.test_option.max_sample_error:
                    self.save_error_code(self.index, code, 'crash')
                else:
                    break
                self.crash += 1
                cur_crash += 1
            elif res.exception:
                if res.except_str.find('op does not support eager execution') != -1:
                    self.terminate('Unsupported: not eager executable.')
                    return
                if res.except_str.find('null session state') != -1:
                    self.terminate('Unsupported: need session.')
                    return
                if res.except_str.find('No such file or directory') != -1 or \
                    res.except_str.find('Found no files at') != -1:
                    self.terminate('Unsupported: need file input.')
                    return
                if res.except_str.find('with arguments from the \'CUDA\' backend') != -1:
                    self.terminate('Unsupported: Func doonot support cuda.')
                    return
                if res.except_str.find('OOM when allocating') != -1:
                    self.OOM += 1
                else:
                    cur_pass = False
                    self.invalid += 1
                    cur_invalid += 1
                    self.log_invalid(res.except_str)
                    if self.test_option.log_invalid:
                        if cur_invalid <= 1:
                            code = '#' + res.except_str.replace('\n', ' ') + '\n' + code
                            self.save_error_code(self.index, code, 'invalid')
            if self.get_cov and self.test_option.record_cov and self.total_times % 10 == 0:
                self.count_cov()
                with open(os.path.join(self.output_dir, 'cov.txt'), 'a') as f:
                    f.write('{}\n'.format(self.coverage))
            if self.test_option.save_non_crash and  \
                    not crash and not timeout:
                self.non_crash += 1
                self.save_error_code(self.index, code, 'non_crash')
            
            if self.save_bmp:
                self.save_bitmap()
                # self.reset_shm()




    def terminate(self, msg=None):
        if self.runner_p.is_alive():
            self.runner_p.kill()
        if self.terminated:
            return
        self.terminated = True
        if self.get_cov:
            self.count_cov()

        if not os.path.exists(os.path.join(self.test_option.output_path, 'res.csv')):
            with open(os.path.join(self.test_option.output_path, 'res.csv'), 'a') as f:
                f.write('api, mode, time, samples, times, invalid, crash, timeout, OOM, pass_rate, coverage\n')
        if msg is not None:
            with open(os.path.join(self.test_option.output_path, 'res.csv'), 'a') as f:
                f.write('{},{},{}\n'.format(self.api_name, self.mode, msg))
        else:
            with open(os.path.join(self.test_option.output_path, 'res.csv'), 'a') as f:
                rate = 1.0 - self.invalid / self.total_times
                f.write('{},{},{:.2f},{},{},{},{},{},{},{:.2%},{}\n'.format(self.api_name, self.mode, time.time() - self.start_time, self.index,\
                    self.total_times, self.invalid, self.crash, self.timeout, self.OOM, rate, self.coverage))

    def log_invalid(self, msg):
        with open(os.path.join(self.output_dir, 'invalid.log'), 'a') as f:
            f.write(self.mode + '\n')
            f.write(msg + '\n')

    def setup_shm(self):
        # map functions
        shmget = self.libc.shmget
        shmat = self.libc.shmat

        shmat.restype = ctypes.c_void_p
        shmat.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)

        # get the shared memory segment

        shmid = shmget(sysv_ipc.IPC_PRIVATE, MAP_SIZE, sysv_ipc.IPC_CREAT | sysv_ipc.IPC_EXCL | 0o600)

        os.environ[SHM_ENV_VAR] = str(shmid)

        # shmid = shmget(sysv_ipc.IPC_PRIVATE, MAP_SIZE,  0o666)
        if shmid < 0:
            sys.exit("cannot get shared memory segment with key %d" % (sysv_ipc.IPC_PRIVATE))

        # map the shared segment into the current process' memory
        # the location (size + 4096) is just a hint
        # shmptr = shmat(shmid, None, 0)
        shmptr = shmat(shmid, None, 0)
        if shmptr == 0 or shmptr== -1:
            sys.exit("cannot attach shared memory segment with id %d" % (shmid))
        self.trace_bits = shmptr
        self.reset_shm()

    def count_cov(self):
        s = ctypes.string_at(self.trace_bits, MAP_SIZE)
        self.coverage = 0
        for i in s:
            if i != 0:
                self.coverage += 1

    def reset_shm(self):
        ctypes.memset(self.trace_bits, 0, MAP_SIZE)
    
    def save_bitmap(self):
        outp = os.path.join(self.output_dir, 'bitmap')
        if not os.path.exists(outp):
            os.mkdir(outp)
        s = ctypes.string_at(self.trace_bits, MAP_SIZE)
        with open(os.path.join(outp, '{}.bmp'.format(self.total_times)), 'wb') as f:
            f.write(s)
