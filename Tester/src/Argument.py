from ArgumentProperty import *
from ArgumentValue import *
from ArgType import ArgType

class Argument:
    def __init__(self, name, prop):
        self.name = name
        self.type = ArgType.from_string(prop['type'])
        self.arg_prop = prop    # ArgumentProperty
        self.init_properties()
        self.arg_value = None   # ArgumentValue
    
    @staticmethod
    def get_base_type_property(name, arg_prop):
        type = ArgType.from_string(arg_prop['type'])
        arg_prop = None
        if type == ArgType.INT:
            arg_prop = IntProperty(name, arg_prop)
        elif type == ArgType.STRING:
            arg_prop = StringProperty(name, arg_prop)
        elif type == ArgType.BOOL:
            arg_prop = BoolProperty(name, arg_prop)
        elif type == ArgType.FLOAT:
            arg_prop = FloatProperty(name, arg_prop)
        elif type == ArgType.TENSOR:
            arg_prop = TensorProperty(name, arg_prop)
        assert(arg_prop != None)
        return arg_prop

    def init_properties(self):
        if self.type == ArgType.INT:
            self.arg_prop = IntProperty(self.name, self.arg_prop)
        elif self.type == ArgType.TYPE:
            self.arg_prop = TypeProperty(self.name, self.arg_prop)
        elif self.type == ArgType.STRING:
            self.arg_prop = StringProperty(self.name, self.arg_prop)
        elif self.type == ArgType.BOOL:
            self.arg_prop = BoolProperty(self.name, self.arg_prop)
        elif self.type == ArgType.FLOAT:
            self.arg_prop = FloatProperty(self.name, self.arg_prop)
        elif self.type == ArgType.TENSOR:
            self.arg_prop = TensorProperty(self.name, self.arg_prop)
        elif self.type == ArgType.LIST:
            self.arg_prop = ListProperty(self.name, self.arg_prop)
        elif self.type == ArgType.DICT:
            self.arg_prop = DictProperty(self.name, self.arg_prop)
        else:
            assert(0)
    
    def generate_random_value(self, value_dict):
        if self.type == ArgType.INT:
            self.arg_value = IntValue(self.arg_prop, value_dict)
        elif self.type == ArgType.TYPE:
            self.arg_value = TypeValue(self.arg_prop, value_dict)
        elif self.type == ArgType.BOOL:
            self.arg_value = BoolValue(self.arg_prop, value_dict)
        elif self.type == ArgType.STRING:
            self.arg_value = StringValue(self.arg_prop, value_dict)
        elif self.type == ArgType.FLOAT:
            self.arg_value = FloatValue(self.arg_prop, value_dict)
        elif self.type == ArgType.TENSOR:
            self.arg_value = TensorValue(self.arg_prop, value_dict)
        elif self.type == ArgType.LIST:
            self.arg_value = ListValue(self.arg_prop, value_dict)
        else:
            assert(0)
        
        if not self.arg_value.generated:
            self.arg_value =None
    
    def clear_value(self):
        self.arg_value = None

    def to_code(self, fw='tf', run_on_cpu=False):
        return self.arg_value.to_code(fw=fw, run_on_cpu=run_on_cpu)