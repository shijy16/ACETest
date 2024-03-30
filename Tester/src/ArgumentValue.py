from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
import re
from random import randint, choice, random, uniform
from ArgType import ArgType

from ArgumentProperty import ArgumentProperty, init_base_type_property

import string

# [min, max]
tf_int = { # 9
    'int8': 'tf.int8', 'int16': 'tf.int16',         # int
    'int': 'tf.int32', 'int32': 'tf.int32',
    'int64': 'tf.int64', 'half': 'tf.half',
    'uint8': 'tf.uint8', 'uint16': 'tf.uint16',     # uint
    'uint': 'tf.uint32', 'uint32': 'tf.uint32',
    'uint64': 'tf.uint64'
}
tf_qint = { # 3
    'qint8': 'tf.qint8', 'qint16': 'tf.qint16',     # qint
    'qint32': 'tf.qint32',
    'quint8': 'tf.quint8', 'quint16': 'tf.quint16',     # quint
}
tf_float = { # 4
    'bfloat16': 'tf.bfloat16', 'float16': 'tf.float16', # float
    'float32': 'tf.float32', 'float64': 'tf.float64',
    'double': 'tf.float64', 'float': 'tf.float32',
}
tf_complex = { # 2
    'complex64': 'tf.complex64', 'complex128': 'tf.complex128', # complex
    'complex': 'tf.complex64', 
}
tf_string = { # 1
    'string': 'tf.string'
}
tf_bool = { # 1
    'bool': 'tf.bool'
}
tf_types = {}
tf_types.update(tf_int)
tf_types.update(tf_qint)
tf_types.update(tf_float)
tf_types.update(tf_complex)
tf_types.update(tf_string)
tf_types.update(tf_bool)

def get_rand_int(min, max):
    if min == max or max < min:
        return min
    return randint(min, max)

def get_rand_bool():
    return bool(randint(0, 1))

def get_rand_string(length):
    res = ''
    for i in range(length):
        c = choice('abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ \t\'!@#$^&*-+/,./<>?;:|`~')
        res += c
    res.replace('\n', ' ')
    res.replace('"', "'")
    return res

def get_rand_float(min, max):
    if prob_bool():
        return min
    elif prob_bool():
        return max
    return uniform(min, max)

def prob_bool(prob=0.1):
    return np.random.choice([True, False], p=[prob, 1.0-prob])

def str2tf_dtype(str_type):
    if str_type == 'any':
        return choice(list(tf_types.values()))
    if str_type in tf_types.values():
        return str_type
    if str_type not in tf_types.keys():
        return 'tf.int32'
    return tf_types[str_type]

def tf_dtype2torch(tf_dtype):
    if tf_dtype in tf_float.values():
        return tf_dtype.replace('tf.', 'torch.')
    elif tf_dtype in tf_int.values():
        torch_dtype = tf_dtype.replace('tf.', 'torch.')
        torch_dtype = torch_dtype.replace('uint', 'int')
        return torch_dtype
    elif tf_dtype in tf_qint.values():
        # no qint16 quint16 in torch
        torch_dtype = tf_dtype.replace('tf.', 'torch.') \
            .replace('qint16', 'qint8')  \
            .replace('quint16', 'quint4x8')
        # Test
        torch_dtype = torch_dtype.replace('qint', 'int') \
            .replace('quint', 'uint').replace('4x8', '16')
        return torch_dtype
    elif tf_dtype in tf_bool.values():
        return tf_dtype.replace('tf.', 'torch.')
    elif tf_dtype in tf_complex.values():
        return tf_dtype.replace('tf.', 'torch.')


class ArgumentValue(ABC):
    large_prob = 0.1
    def __init__(self, arg_prop):
        self.idx_in_list = arg_prop.idx_in_list
        self.name = arg_prop.name
        self.generated = False
        self.default = arg_prop.default
        if arg_prop.use_default and self.default is not None and arg_prop.value is None:
            self.value = self.default
            self.generated = True
    
    @abstractmethod
    def to_code(self, fw='tf', run_on_cpu=False):
        pass

    def solve_constraint(self, line, value_dict, generator=None):
        if line == '&NONE':
            return '&NONE'
        result = None
        pattern = re.compile(r'&[a-zA-Z0-9_]+')
        res = re.search(pattern, line)
        cnt = 0
        while(res):
            cnt += 1
            if cnt > 1000:
                print(line)
                return None
            dep_arg = res.group(0)[1:]
            if dep_arg == 'ANY':
                line = line.replace(res.group(0), str(eval(generator)))
            elif dep_arg == 'INDEX':
                assert(self.idx_in_list is not None)
                line = line.replace(res.group(0), str(self.idx_in_list))
            elif dep_arg == 'INT_BETWEEN':
                line = line.replace(res.group(0), 'get_rand_int')
            elif dep_arg in value_dict.keys():
                line = line.replace(res.group(0), 'value_dict[\'{}\']'.format(dep_arg))
            else:
                return None
            res = re.search(pattern, line)
        result = eval(line)
        return result
    

    def get_with_type(self, target_type, prop, value_dict, generator=None):
        if prop is None:
            return eval(generator)
        elif type(prop) == target_type and type(prop) != str: # handle str individually
            return deepcopy(prop)
        elif (type(prop) in [float, int]) and target_type in [float, int]:
            return deepcopy(prop)
        elif type(prop) == str and prop.find('&') > -1:
            return self.solve_constraint(prop, value_dict, generator)
        elif type(prop) == str: # string value without constraint in it
            return prop
        print(type(prop), prop, target_type)
        assert(0)

    def get_int(self, prop, value_dict, generator=None):
        return self.get_with_type(int, prop, value_dict, generator)

    def get_bool(self, prop, value_dict, generator=None):
        if type(prop) is int:
            if prop == 0:
                prop = False
            elif prop == 1:
                prop = True
            else:
                assert(0)
        return self.get_with_type(bool, prop, value_dict, generator)

    def get_list(self, prop, value_dict, generator=None):
        return self.get_with_type(list, prop, value_dict, generator)

    def get_shape(self, prop, value_dict, ndim):
        if not prop:
            shape = [get_rand_int(0, 16) for i in range(ndim)]
            if ndim != 0:
                if prob_bool(self.large_prob):
                    shape[get_rand_int(0, len(shape) - 1)] = get_rand_int(2**32, 2**64-1)
            return shape
        if type(prop) == str:
            prop = self.get_list(prop, value_dict)
            if prop is None:
                return None
        assert(type(prop) == list)
        res = []
        for element in prop:
            if type(element) == IntValue:
                res.append(element.value)
                if element.value is None:
                    return None
            else:
                res.append(self.get_with_type(int, element, value_dict, 'get_rand_int(0, 32)'))
        return res

    def get_constraint_list(self, prop, value_dict, child_type=int, child_generator=None):
        assert(prop is not None)
        res = []
        for element in prop:
            res.append(self.get_with_type(child_type, element, value_dict, child_generator))
        return res

    def get_string(self, prop, value_dict, generator=None):
        return self.get_with_type(str, prop, value_dict, generator)
    
    def get_float(self, prop, value_dict, generator=None):
        return self.get_with_type(float, prop, value_dict, generator)

    def get_dtype(self, prop, value_dict, default_value='tf.float32'):
        if not prop:
            return choice(list(tf_types.values()))
        if type(prop) == list:
            if 'dtype' in prop:
                prop.remove('dtype')
            if len(prop) == 0:
                return default_value
            return str2tf_dtype(choice(prop))
        elif type(prop) == str and prop.find('&') > -1:
            dtype = self.solve_constraint(prop, value_dict)
            if dtype is None:
                return default_value
            elif type(dtype) == str:
                return str2tf_dtype(dtype)
            else:
                return dtype
        elif type(prop) == str:
            return str2tf_dtype(prop)
        assert(0)

    def get_type(self, prop, value_dict, default_value=ArgType.INT):
        if not prop:
            return default_value
        if type(prop) == list:
            return ArgType.from_string(choice(prop))
        elif type(prop) == str and prop.find('&') > -1:
            t = self.solve_constraint(prop, value_dict)
            if t is None:
                return None
            elif type(t) == str:
                return ArgType.from_string(t)
            else:
                if t in tf_int.values() or t in tf_qint.values():
                    return ArgType.INT
                elif t in tf_float.values():
                    return ArgType.FLOAT
                elif t in tf_bool.values():
                    return ArgType.BOOL
                elif t in tf_string.values():
                    return ArgType.STRING
                return t
        elif type(prop) == str:
            return ArgType.from_string(prop)
        assert(0)

class TypeValue(ArgumentValue):
    def __init__(self, arg_prop, value_dict):
        super().__init__(arg_prop)
        self.dtype = 'type'
        if self.generated:
            self.value = str2tf_dtype(self.value)
            return

        if arg_prop.enum is not None:
            self.value = str2tf_dtype(choice(arg_prop.enum))
        else:
            self.value = str2tf_dtype(choice(list(tf_types.keys())))
        self.generated = True

    def to_code(self, fw='tf', run_on_cpu=False):
        if fw == 'torch':
            self.value = tf_dtype2torch(self.value)
        return '{} = {}'.format(self.name, self.value)

class IntValue(ArgumentValue):
    def __init__(self, arg_prop, value_dict):
        super().__init__(arg_prop)
        self.dtype = 'int'
        if self.generated:
            return
        
        self.enum = self.get_list(arg_prop.enum, value_dict, '[]')
        if self.enum is None:
            return

        if len(self.enum) > 0:
            self.value = choice(self.enum)
            self.generated = True
            return

        if self.name == 'N' and arg_prop.max is None:
            arg_prop.max = 6
        

        self.max = self.get_int(arg_prop.max, value_dict, '2**63-1 if prob_bool(self.large_prob) else 128')
        if self.max is None:
            return
        elif self.max > 128:
            if not prob_bool():
                self.max = 128

        self.min = self.get_int(arg_prop.min, value_dict, '-2**63 if prob_bool(self.large_prob) else -128')
        if self.min is None:
            return
        elif self.min < -128:
            if not prob_bool():
                self.min = -128
        
        self.value = self.get_int(arg_prop.value, value_dict, 'get_rand_int({}, {})'.format(self.min, self.max))
        if self.value is None:
            return
        if prob_bool(0.01):
            self.value = 0
        if prob_bool(self.large_prob):
            self.value = randint(-2**64+1, 2**64-1)

        self.generated = True

    def to_code(self, fw='tf', run_on_cpu=False):
        return '{} = {}'.format(self.name, self.value)

class BoolValue(ArgumentValue):
    def __init__(self, arg_prop, value_dict):
        super().__init__(arg_prop)
        self.dtype = 'bool'
        if self.generated:
            return

        if self.name == 'use_cudnn_on_gpu':
            self.value = True
            self.generated = True
            return

        self.value = self.get_bool(arg_prop.value, value_dict, 'get_rand_bool()')
        if self.value is None:
            return
        self.generated = True

    def to_code(self, fw='tf', run_on_cpu=False):
        return '{} = {}'.format(self.name, self.value)


class StringValue(ArgumentValue):
    def __init__(self, arg_prop, value_dict):
        super().__init__(arg_prop)
        self.is_byte = False
        self.dtype = 'string'
        if self.generated:
            return

        if arg_prop.value is None:
            # enum  -> length -> value
            self.enum = self.get_list(arg_prop.enum, value_dict, '[]')
            if self.enum is None:
                return

            if len(self.enum) > 0:
                self.value = choice(self.enum)
                self.generated = True
                return

        self.length = self.get_int(arg_prop.length, value_dict, 'get_rand_int(0, 300)')
        
        if self.length is None:
            return
        self.value = self.get_string(arg_prop.value, value_dict, 'get_rand_string({})'.format(self.length))
        if self.value is None:
            return
        if arg_prop.img_fmt is not None:
            formats = self.get_list(arg_prop.img_fmt, value_dict, '[]')
            if len(formats) > 0:
                fmt = choice(formats)
                self.is_byte = True
                if fmt == 'JPEG':
                    self.value = b'\xff\xd8\xff' + self.value.encode('utf-8')
                elif fmt == 'BMP':
                    self.value = b'\x42\x4d' + self.value.encode('utf-8')
                elif fmt == 'PNG':
                    self.value = b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A' + self.value.encode('utf-8')
                elif fmt == 'GIF':
                    self.value = b'\x47\x49\x46\x38"' + self.value.encode('utf-8')
    
        self.generated = True

    def to_code(self, fw='tf', run_on_cpu=False):
        if self.is_byte:
            return '{} = {}'.format(self.name, self.value)
        return '{} = "{}"'.format(self.name, self.value)

class FloatValue(ArgumentValue):
    def __init__(self, arg_prop, value_dict):
        super().__init__(arg_prop)
        self.dtype = 'float'
        if self.generated:
            return

        self.max = self.get_float(arg_prop.max, value_dict, '2e64 if prob_bool(self.large_prob) else 128.0')
        if self.max is None:
            return

        self.min = self.get_float(arg_prop.min, value_dict, '-2e64 if prob_bool(self.large_prob) else -128.0')
        if self.min is None:
            return

        self.value = self.get_float(arg_prop.value, value_dict, 'get_rand_float({}, {})'.format(self.min, self.max))
        if self.value is None:
            return

        if prob_bool(0.01):
            self.value = 0
        if prob_bool(self.large_prob):
            self.value = randint(-2**64+1, 2**64-1)

        self.generated = True

    def to_code(self, fw='tf', run_on_cpu=False):
        return '{} = {}'.format(self.name, self.value)

class TensorValue(ArgumentValue):
    def __init__(self, arg_prop, value_dict):
        super().__init__(arg_prop)
        if self.generated:
            return

        self.dtype = self.get_dtype(arg_prop.dtype, value_dict)
        if self.dtype is None:
            return
        
        self.min_ndim = self.get_int(arg_prop.min_ndim, value_dict, '0')
        if self.min_ndim is None:
            return

        self.max_ndim = self.get_int(arg_prop.max_ndim, value_dict, '6')
        if self.max_ndim is None:
            return

        if self.dtype in tf_string.values() and arg_prop.ndim == 0:
            arg_prop.ndim = None

        self.ndim = self.get_int(arg_prop.ndim, value_dict, 'randint({}, {})'.format(self.min_ndim, self.max_ndim))
        if self.ndim is None:
            return

        self.shape = self.get_shape(arg_prop.shape, value_dict, self.ndim)
        if self.shape is None:
            return

        for i in range(len(self.shape)):
            if self.shape[i] is None:
                return
            else:
                self.shape[i] = int(self.shape[i])

        self.num_elements = 1
        for i in self.shape:
            self.num_elements *= i

        self.max_val = 2 ** 10
        self.min_val = - 2 ** 10
        if self.dtype.find('int8') != -1: 
            self.max_val = 2 ** 7
            self.min_val = -2 ** 7
        if prob_bool(0.01):
            self.min_val = 0
            self.max_val = 0
        if prob_bool(self.large_prob):
            self.max_val = 2 ** 64 -1
            self.min_val = - 2 ** 64 + 1
        

        self.value = None
        if arg_prop.value is not None:
            self.value = self.get_constraint_list(arg_prop.value, value_dict, child_type=int, child_generator='get_rand_int(1, 128)')
            if self.value is None:
                return

        self.generated = True

    def to_code(self, fw='tf', run_on_cpu=False):
        if self.value is not None:
            return '{} = {}'.format(self.name, self.value)
        else:
            if fw == 'tf':
                code = ""
                if self.dtype in tf_float.values():
                    code += "%s = tf.random.uniform(%s, dtype=%s, minval=%d, maxval=%d)\n" % \
                        (self.name, self.shape, self.dtype, self.min_val, self.max_val)
                elif self.dtype in tf_complex.values():
                    ftype = 'tf.float64' if self.dtype == 'tf.complex128' else 'tf.float32'
                    code += "%s = tf.complex(tf.random.uniform(%s, dtype=%s, minval=%d, maxval=%d)," \
                            "tf.random.uniform(%s, dtype=%s, minval=%d, maxval=%d))\n" % \
                                (self.name, self.shape, ftype, self.min_val, self.max_val, self.shape, ftype, self.min_val, self.max_val)
                elif self.dtype in tf_bool.values():
                    code += "%s = tf.cast(tf.random.uniform(" \
                        "%s, minval=0, maxval=2, dtype=tf.int32), dtype=tf.bool)\n" % (self.name, self.shape)
                elif self.dtype in tf_qint.values():
                    code += "%s = tf.cast(tf.random.uniform(" \
                        "%s, minval=0, maxval=64, dtype=tf.int64), dtype=%s)\n" % (self.name, self.shape, self.dtype)
                elif self.dtype in tf_string.values():
                    if self.default is not None:
                        code += "%s = '%s'" % (self.name, self.default)
                    else:
                        temp_shape = self.shape
                        # add string length as last dim
                        temp_shape.append(get_rand_int(0, 300))
                        code += "%s = tf.strings.unicode_encode(np.random.randint(0, 128, size=%s, dtype=np.int32), output_encoding='UTF-8')\n" \
                            % (self.name, temp_shape)
                else:
                    code += "%s = tf.saturate_cast(" \
                        "tf.random.uniform(%s, minval=%d, maxval=%d, dtype=tf.int64), " \
                        "dtype=%s)\n" % (self.name, self.shape, self.min_val, self.max_val, self.dtype)
            elif fw == 'torch':
                post_fix = ''
                if not run_on_cpu:
                    post_fix = '.cuda()'
                code = ""
                # self.dtype = 'tf.int32'
                if self.dtype in tf_float.values() or self.dtype in tf_complex.values():
                    dtype = tf_dtype2torch(self.dtype)
                    code += "%s = torch.rand(%s, dtype=%s)%s\n" % (self.name, self.shape, dtype, post_fix)
                elif self.dtype in tf_bool.values():
                    dtype = tf_dtype2torch(self.dtype)
                    code += "%s = torch.randint(0, 2, %s, dtype=%s)%s\n" % (self.name, self.shape, dtype, post_fix)
                elif self.dtype in tf_int.values() or self.dtype in tf_qint.values():
                    dtype = tf_dtype2torch(self.dtype)
                    code += "%s = torch.randint(%d, %d, %s, dtype=%s)%s\n" % (self.name, self.min_val, self.max_val, self.shape, dtype, post_fix)
                # There is no torch.string in pytorch
                elif self.dtype in tf_string.values():
                    code += "%s = torch.randint(%d, %d, %s, dtype=%s)%s\n" % (self.name, self.min_val, self.max_val, self.shape, 'torch.int32', post_fix)
                else:
                    print(self.dtype)
                    assert(0)
        return code

class ListValue(ArgumentValue):
    def __init__(self, arg_prop, value_dict):
        super().__init__(arg_prop)
        if self.generated:
            return
        
        self.min_len = self.get_int(arg_prop.min_len, value_dict, '0')
        if self.min_len is None:
            return
        
        self.max_len = self.get_int(arg_prop.max_len, value_dict, '6')
        if self.max_len is None:
            return
        
        self.length = self.get_int(arg_prop.length, value_dict, 'randint({}, {})'.format(self.min_len, self.max_len))
        if self.length is None:
            return
        elif self.length > 2**10:
            self.length = 2**10
        elif self.length < 0:
            self.length = 0

        self.value = self.get_list(arg_prop.value, value_dict, '[None]*{}'.format(self.length))
        if type(arg_prop.child_prop) is list:
            child_prop = choice(arg_prop.child_prop)
        else:
            child_prop = arg_prop.child_prop
        if child_prop is None:
            child_prop = {}

        self.dtype = 'int'
        for i in range(self.length):
            temp = deepcopy(child_prop)
            temp['idx_in_list'] = i
            if self.value[i] != None:
                temp['value'] = self.value[i]
            arg_value = self.get_value_by_prop(temp, value_dict)
            if arg_value is None or not arg_value.generated:
                return
            self.value[i] = arg_value

        self.generated = True

    def get_value_by_prop(self, arg_prop, value_dict):
        if 'type'  not in arg_prop.keys():
            arg_prop['type'] = 'int'
            arg_prop['min'] = 0
        type = self.get_type(arg_prop['type'], value_dict)
        if type is None:
            return None
        arg_value = None
        if type == ArgType.INT:
            arg_prop = init_base_type_property(None, type, arg_prop)
            arg_value = IntValue(arg_prop, value_dict)
            self.dtype = 'int'
        elif type == ArgType.BOOL:
            arg_prop = init_base_type_property(None, type, arg_prop)
            arg_value = BoolValue(arg_prop, value_dict)
            self.dtype = 'bool'
        elif type == ArgType.STRING:
            arg_prop = init_base_type_property(None, type, arg_prop)
            arg_value = StringValue(arg_prop, value_dict)
            self.dtype = 'string'
        elif type == ArgType.FLOAT:
            arg_prop = init_base_type_property(None, type, arg_prop)
            arg_value = FloatValue(arg_prop, value_dict)
            self.dtype = 'float'
        elif type == ArgType.TENSOR:
            arg_prop = init_base_type_property(None, type, arg_prop)
            arg_value = TensorValue(arg_prop, value_dict)
            self.dtype = 'tensor'
        elif type == ArgType.LIST:
            arg_prop = init_base_type_property(None, type, arg_prop)
            arg_value = ListValue(arg_prop, value_dict)
            self.dtype = 'list'
        elif type == ArgType.TYPE:
            arg_prop = init_base_type_property(None, type, arg_prop)
            arg_value = TypeValue(arg_prop, value_dict)
            self.dtype = 'type'
        else:
            print(type, arg_prop['type'])
            assert(0)
        return arg_value

    def to_code(self, fw='tf', run_on_cpu=False):
        code = ''
        args_line = ''
        for i in range(len(self.value)):
            cur_name = self.name + '_' + str(i)
            if not isinstance(self.value[i], ArgumentValue):
                args_line += str(self.value[i]) + ', '
            else:
                self.value[i].name = cur_name
                args_line += cur_name + ', '
                if type(self.value[i]) in [int, float, str]:
                    code += '{} = {}\n'.format(cur_name, self.value[i])
                else:
                    code += self.value[i].to_code(fw, run_on_cpu) + '\n'
        last_line = '{} = [{}]'.format(self.name, args_line)
        code += last_line
        return code
