import os
class TestOption:
    def __init__(self, fw='tf', mode='all', filter=None, use_cons=True, test_round=2000, round_timeout=10, api_timeout=1800, total_round=1, target_api=None, log_invalid=True, save_non_crash=False, get_cov=False, work_path=None):
        self.fw = fw
        self.mode = mode
        self.filter = filter
        self.use_cons = use_cons
        self.test_round = test_round
        self.round_timeout = round_timeout
        self.api_timeout = api_timeout
        self.target_api = target_api
        self.log_invalid = log_invalid
        self.save_non_crash = save_non_crash
        self.p_num = 24
        self.max_api_error = 50
        self.max_sample_error = 5
        self.total_round = total_round
        self.work_path = work_path

        # coverage
        self.get_cov = get_cov
        self.save_bitmap = False
        self.record_cov = False
    
    def initialize(self, round=0):
        assert(self.filter in ['after', 'new', 'all', 'exist', 'list', None])
        assert(self.mode in ['gpu', 'cpu_ori', 'cpu_ondednn', 'cov', 'all'])
        assert(self.fw in ['tf', 'torch'])
        if self.fw == 'torch':
            assert(self.mode != 'cpu_onednn')
        constraint_dir = {
            'tf' : '../data/tensorflow/constraints',
            'torch' : '../data/pytorch/constraints'
        }
        self.constraint_dir = constraint_dir[self.fw]
        api2op_csv = {
            'tf' : '../data/tensorflow/API2OP.csv',
            'torch' : '../data/pytorch/API2OP.csv'
        }
        self.api2op_csv = api2op_csv[self.fw]
        api_dir = {
            'tf' : '../data/tensorflow/op_json',
            'torch' : '../data/pytorch/op_json'
        }
        self.api_dir = api_dir[self.fw]
        self.sampler = './smtsampler'
        self.output_path = 'output_' + self.fw + '_' + str(round)
        if self.work_path:
            if not os.path.exists(self.work_path):
                os.mkdir(self.work_path)
            self.output_path = os.path.join(self.work_path, self.output_path)
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
