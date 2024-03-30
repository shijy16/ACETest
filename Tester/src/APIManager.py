from ArgType import ArgType
from Option import TestOption
import numpy as np
from ArgumentValue import ArgumentValue

def add_indent(code):
    lines = code.split('\n')
    res = ''
    for line in lines:
        if len(line) == 0:
            continue
        res += '    ' + line + '\n'
    return res

def prob_bool(prob=0.1):
    return np.random.choice([True, False], p=[prob, 1.0-prob])

class APIManager:
    def __init__(self, api_name, args, test_option : TestOption):
        self.api_name = api_name
        self.arg_list = args
        self.test_option = test_option
    
    def generate_random_args(self):
        if prob_bool(0.01):
            ArgumentValue.large_prob = 1
        else:
            ArgumentValue.large_prob = 0.01
        all_generated = False
        value_dict = {}
        for arg in self.arg_list:
            arg.clear_value()

        loop_cnt = 0
        while not all_generated:
            if loop_cnt > 1000:
                print(value_dict)
                assert(0)
            loop_cnt += 1
            all_generated = True
            for arg in self.arg_list:
                if not arg.arg_value:
                    arg.generate_random_value(value_dict)
                if not arg.arg_value:
                    all_generated = False
                else:
                    value_dict[arg.name] = arg.arg_value

    # Generate call codes.
    '''
        res = tf.raw_ops.xxx(
            arg1 = ...,
            arg2 = ...,
        )
    '''
    def generate_call_code(self, run_on_cpu=False, fw='tf'):
        args_code = ''
        if len(self.arg_list) == 1 and fw == 'torch':
                args_code += '{}\n'.format(self.arg_list[0].name)
        else:
            for arg in self.arg_list:
                if arg.arg_prop.is_mid:
                    continue
                # if arg.arg_prop.type == ArgType.TYPE and arg.name.startswith('T'):
                #     continue
                args_code += '{}={},\n'.format(arg.name, arg.name)
        args_code = add_indent(args_code)
        return '{} = {}(\n{})'.format('res', self.api_name, args_code)
    


    # Generate main script.
    def to_tf_code(self, run_on_cpu=False):
        code = 'import tensorflow as tf\n'
        code += 'import numpy as np\n'
        code += 'try:\n'

        device = 'CPU' if run_on_cpu else 'GPU:2'
        device_content = 'with tf.device("{}"):\n'.format(device)

        content = ''
        for arg in self.arg_list:
            if arg.arg_prop.is_mid:
                continue
            # if arg.arg_prop.type == ArgType.TYPE and arg.name.startswith('T'):
            #     continue
            content += arg.to_code(fw='tf', run_on_cpu=run_on_cpu) + '\n'
        content += self.generate_call_code(run_on_cpu=run_on_cpu, fw='tf')
        
        device_content += add_indent(content)
        code += add_indent(device_content)

        code += 'except Exception as e:\n'
        code += add_indent('exception_str = str(e)\n')
        code += add_indent('ends_with_exception = True\n')
        code = code.replace("'&NONE'", 'None').replace('&NONE', 'None')
        return code
    
    def to_torch_code(self, run_on_cpu=False):
        code = ''
        code += 'import torch\n'
        code += 'import numpy as np\n'
        code += 'try:\n'

        content = ''
        for arg in self.arg_list:
            if arg.arg_prop.is_mid:
                continue
            if arg.arg_prop.type == ArgType.TYPE and arg.name.startswith('T'):
                continue
            content += arg.to_code(fw='torch', run_on_cpu=run_on_cpu) + '\n'
        
        content += self.generate_call_code(run_on_cpu=run_on_cpu, fw='torch')
        
        code += add_indent(content)

        code += 'except Exception as e:\n'
        code += add_indent('exception_str = str(e)\n')
        code += add_indent('ends_with_exception = True\n')
        code = code.replace("'&NONE'", 'None').replace('&NONE', 'None')
        return code

    
    def to_code(self, run_on_cpu=False):
        if self.test_option.fw == 'tf':
            return self.to_tf_code(run_on_cpu)
        if self.test_option.fw == 'torch':
            return self.to_torch_code(run_on_cpu)

    

    def get_cpu_code(self):
        return self.to_code(run_on_cpu=True)
    
    
    def get_gpu_code(self):
        return self.to_code(run_on_cpu=False)
    
    def to_dict(self):
        res = []
        for arg in self.arg_list:
            res.append(arg.arg_prop.to_dict())
        return res
