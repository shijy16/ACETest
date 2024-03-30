from enum import IntEnum

inttype = [
    'int8', 'int16', 'int', 'int32', 'int64',
    'half', 'uint8', 'uint16', 'uint', 'uint32', 'uint64',
]

floattype = [
    'bfloat16', 'float16', 'float32', 'float64', 'float'
]

class ArgType(IntEnum):
    INT = 1
    STRING = 2
    BOOL = 3
    FLOAT = 4
    TENSOR = 5
    LIST = 6
    DICT = 7
    TYPE = 8

    @staticmethod
    def from_string(type):
        if type.lower() in inttype:
            return ArgType.INT
        elif type.lower() == 'string':
            return ArgType.STRING
        elif type.lower() == 'bool':
            return ArgType.BOOL
        elif type.lower() in floattype or type.lower() == 'double':
            return ArgType.FLOAT
        elif type.lower() == 'tensor':
            return ArgType.TENSOR
        elif type.lower() == 'list' or type.lower() == 'shape':
            return ArgType.LIST
        elif type.lower() == 'dict':
            return ArgType.DICT
        elif type.lower() == 'type':
            return ArgType.TYPE
        print('Error: Unknown type:', type)
        assert(0)