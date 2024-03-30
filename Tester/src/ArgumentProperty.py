from abc import ABC, abstractmethod
from ArgType import ArgType

class ArgumentProperty(ABC):
    def __init__(self, prop, use_default=False):
        self.idx_in_list = None
        self.is_attr = self.get_in_dict('is_attr', prop)
        self.default = self.get_in_dict('default', prop)
        self.is_mid = self.get_in_dict('is_mid', prop)
        self.value = self.get_in_dict('value', prop)
        self.input_order = self.get_in_dict('input_order', prop)
        self.optional = self.get_in_dict('optional', prop)
        self.use_default = use_default


    @staticmethod
    def get_in_dict(name, prop):
        if name in prop.keys():
            return prop[name]
        return None

    @abstractmethod
    def to_dict(self):
        pass


def init_base_type_property(name, type, prop):
    arg_prop = None
    if type == ArgType.INT:
        arg_prop = IntProperty(name, prop)
    elif type == ArgType.STRING:
        arg_prop = StringProperty(name, prop)
    elif type == ArgType.BOOL:
        arg_prop = BoolProperty(name, prop)
    elif type == ArgType.FLOAT:
        arg_prop = FloatProperty(name, prop)
    elif type == ArgType.TENSOR:
        arg_prop = TensorProperty(name, prop)
    elif type == ArgType.LIST:
        arg_prop = ListProperty(name, prop)
    elif type == ArgType.TYPE:
        arg_prop = TypeProperty(name, prop)
    assert(arg_prop != None)
    return arg_prop

class IntProperty(ArgumentProperty):
    def __init__(self, name, prop):
        super().__init__(prop)
        self.type = ArgType.INT
        self.name = name
        self.max = self.get_in_dict('max', prop)
        self.min = self.get_in_dict('min', prop)
        self.enum = self.get_in_dict('enum', prop)
    
    def to_dict(self):
        return {
            'is_attr': self.is_attr,
            'input_order': self.input_order,
            'default': self.default, 
            'is_mid': self.is_mid,
            'name' : self.name,
            'type' : 'int',
            'max' : self.max,
            'min' : self.min,
            'value' : self.value,
            'enum' : self.enum,
            'optional' : self.optional
        }


class TypeProperty(ArgumentProperty):
    def __init__(self, name, prop):
        super().__init__(prop)
        self.type = ArgType.TYPE
        self.name = name
        self.enum = self.get_in_dict('enum', prop)

    def to_dict(self):
        return {
            'is_attr': self.is_attr,
            'input_order': self.input_order,
            'default': self.default, 
            'is_mid': self.is_mid,
            'name' : self.name,
            'type' : 'type',
            'enum' : self.enum,
            'optional' : self.optional
        }

class StringProperty(ArgumentProperty):
    def __init__(self, name, prop):
        super().__init__(prop)
        self.type = ArgType.STRING
        self.name = name
        self.enum = self.get_in_dict('enum', prop)
        self.length = self.get_in_dict('length', prop)
        self.max_len = self.get_in_dict('max_len', prop)
        self.min_len = self.get_in_dict('min_len', prop)
        self.img_fmt = self.get_in_dict('img_fmt', prop)

    def to_dict(self):
        return {
            'is_attr': self.is_attr,
            'input_order': self.input_order,
            'default': self.default, 
            'is_mid': self.is_mid,
            'name' : self.name,
            'type' : 'string',
            'value' : self.value,
            'enum' : self.enum,
            'length' : self.length,
            'max_len' : self.max_len,
            'min_len' : self.min_len,
            'optional' : self.optional,
            'img_fmt': self.img_fmt
        }


class BoolProperty(ArgumentProperty):
    def __init__(self, name, prop):
        super().__init__(prop)
        self.type = ArgType.BOOL
        self.name = name

    def to_dict(self):
        return {
            'is_attr': self.is_attr,
            'input_order': self.input_order,
            'default': self.default, 
            'is_mid': self.is_mid,
            'name' : self.name,
            'type' : 'bool',
            'value' : self.value,
            'optional' : self.optional
        }


class FloatProperty(ArgumentProperty):
    def __init__(self, name, prop):
        super().__init__(prop)
        self.type = ArgType.FLOAT
        self.name = name
        self.max = self.get_in_dict('max', prop)
        self.min = self.get_in_dict('min', prop)

    def to_dict(self):
        return {
            'is_attr': self.is_attr,
            'input_order': self.input_order,
            'default': self.default, 
            'is_mid': self.is_mid,
            'name' : self.name,
            'type' : 'float',
            'max' : self.max,
            'min' : self.min,
            'value' : self.value,
            'optional' : self.optional
        }



class TensorProperty(ArgumentProperty):
    def __init__(self, name, prop):
        super().__init__(prop)
        self.type = ArgType.TENSOR
        self.name = name
        self.shape = self.get_in_dict('shape', prop)
        self.ndim = self.get_in_dict('ndim', prop)
        self.min_ndim = self.get_in_dict('min_ndim', prop)
        self.max_ndim = self.get_in_dict('max_ndim', prop)
        self.dtype = self.get_in_dict('dtype', prop)
    
    def to_list(self):
        list_prop_dict = {
            'name' : self.name,
            'input_order': self.input_order,
            'min_len': self.min_ndim,
            'max_len': self.max_ndim,
            'length': self.ndim,
            'value': self.shape,
            'child_prop' : {}
        }
        if type(self.dtype) is list:
            self.dtype = self.dtype[0]
        elif self.dtype.find('float') != -1 or self.dtype.find('double') != -1:
            list_prop_dict['child_prop']['type'] = 'float'
        elif self.dtype.find('string') != -1:
            list_prop_dict['child_prop']['type'] = 'string'
        else:
            list_prop_dict['child_prop']['type'] = 'int'
        return ListProperty(self.name, list_prop_dict)

    
    def to_scalar(self):
        scalar_prop_dict = {
            'name' : self.name,
            'input_order': self.input_order
        }

        if type(self.dtype) is list:
            assert(0)
        if self.dtype is None:
            scalar_prop_dict['type'] = 'int'
            return IntProperty(self.name, scalar_prop_dict)
        elif self.dtype.find('float') != -1 or self.dtype.find('double') != -1:
            scalar_prop_dict['type'] = 'float'
            return FloatProperty(self.name, scalar_prop_dict)
        elif self.dtype.find('string') != -1:
            scalar_prop_dict['type'] = 'string'
            return StringProperty(self.name, scalar_prop_dict)
        else:
            scalar_prop_dict['type'] = 'int'
            return IntProperty(self.name, scalar_prop_dict)

    def to_dict(self):
        d = {
            'input_order': self.input_order,
            'is_attr': self.is_attr,
            'default': self.default, 
            'is_mid': self.is_mid,
            'name' : self.name,
            'type' : 'tensor',
            'value' : self.value,
            'shape' : self.shape,
            'ndim' : self.ndim,
            'max_ndim' : self.max_ndim,
            'min_ndim' : self.min_ndim,
            'dtype' : self.dtype,
            'optional' : self.optional
        }
        return d


class ListProperty(ArgumentProperty):
    def __init__(self, name, prop):
        super().__init__(prop)
        self.type = ArgType.LIST
        self.name = name
        self.length = self.get_in_dict('length', prop)
        self.child_prop = self.get_in_dict('child_prop', prop)
        if self.child_prop is None:
            self.child_prop = {}
        self.max_len = self.get_in_dict('max_len', prop)
        self.min_len = self.get_in_dict('min_len', prop)
        # if self.child_prop is not None:
        #     if type(self.child_prop) is list:
        #         for i in range(len(self.child_prop)):
        #             self.child_prop[i] = init_base_type_property(self.name + '_child', prop=self.child_prop[i]) 
        #     else:
        #         self.child_prop = init_base_type_property(self.name + '_child', prop=self.child_prop)

    def to_dict(self):
        return {
            'is_attr': self.is_attr,
            'input_order': self.input_order,
            'default': self.default, 
            'is_mid': self.is_mid,
            'name' : self.name,
            'type' : 'list',
            'value' : self.value,
            'length' : self.length,
            'max_len' : self.max_len,
            'min_len' : self.min_len,
            'child_prop' : self.child_prop,
            'optional' : self.optional
        }

    

class DictProperty(ArgumentProperty):
    def __init__(self, name, prop):
        super().__init__(prop)
        self.type = ArgType.DICT
        self.name = name
        self.ktype = self.get_in_dict('ktype', prop)
        self.vtype = self.get_in_dict('vtype', prop)
