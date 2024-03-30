import z3
import os
import re
import hashlib

from enum import IntEnum

class VarType(IntEnum):
    VAR = 0
    EXTEND_VAR = 1
    ARG = 2
    UNINTERESTING = 3

    STR_FMT = 10
    STR_LEN = 11
    
    LIST_VAL = 20
    LIST_LEN = 21
    LIST_ITERATOR_VAL = 22
    LIST_MIN = 23
    LIST_MAX = 24

    TENSOR_DIM = 30
    TENSOR_DIMSIZE = 31
    TENSOR_NUM_ELEMENT = 32
    TENSOR_MIN_DIM = 33

    @staticmethod
    def get_var_type(var_name):
        res = re.match(r'var([0-9]+)$', str(var_name))
        if res is not None:
            return int(res.group(1)), VarType.VAR

        res = re.match(r'temp([0-9]+)_([0-9]+)(_[A-Z]+)?$', str(var_name))
        if res is not None:
            return int(res.group(1)), VarType.VAR

        res = re.match(r'extend_([0-9]+)[A-Z]*$', str(var_name))
        if res is not None:
            return int(res.group(1)), VarType.EXTEND_VAR

        res = re.match(r'ARGNAME_(.*)_NAMEEND_STRFMT$', str(var_name))
        if res is not None:
            return res.group(1), VarType.STR_FMT

        res = re.match(r'ARGNAME_(.*)_NAMEEND_VAL$', str(var_name))
        if res is not None:
            return res.group(1), VarType.LIST_VAL

        res = re.match(r'ARGNAME_(.*)_NAMEEND_LIST_MIN$', str(var_name))
        if res is not None:
            return res.group(1), VarType.LIST_MIN
        
        res = re.match(r'ARGNAME_(.*)_NAMEEND_LIST_MAX$', str(var_name))
        if res is not None:
            return res.group(1), VarType.LIST_MAX

        res = re.match(r'ARGNAME_(.*)_NAMEEND_LEN$', str(var_name))
        if res is not None:
            return res.group(1), VarType.LIST_LEN

        res = re.match(r'ARGNAME_(.*)_NAMEEND_ITERATOR_VAL$', str(var_name))
        if res is not None:
            return res.group(1), VarType.LIST_ITERATOR_VAL

        res = re.match(r'ARGNAME_(.*)_NAMEEND_DIMSIZE$', str(var_name))
        if res is not None:
            return res.group(1), VarType.TENSOR_DIMSIZE

        res = re.match(r'ARGNAME_(.*)_NAMEEND_DIM$', str(var_name))
        if res is not None:
            return res.group(1), VarType.TENSOR_DIM

        res = re.match(r'ARGNAME_(.*)_NAMEEND_NUMELEMENT$', str(var_name))
        if res is not None:
            return res.group(1), VarType.TENSOR_NUM_ELEMENT

        res = re.match(r'ARGNAME_(.*)_NAMEEND_MINDIM$', str(var_name))
        if res is not None:
            return res.group(1), VarType.TENSOR_MIN_DIM

        res = re.match(r'ARGNAME_(.*)_NAMEEND$', str(var_name))
        if res is not None:
            return res.group(1), VarType.ARG
        
        # print('Error: Unknown', var_name)
        return 'None', VarType.VAR

# Wrapper for allowing Z3 ASTs to be stored into Python Hashtables. 
class AstRefKey:
    def __init__(self, n):
        self.n = n
    def __hash__(self):
        return self.n.hash()
    def __eq__(self, other):
        return self.n.eq(other.n)
    def __repr__(self):
        return str(self.n)

def askey(n):
    assert isinstance(n, z3.AstRef)
    return AstRefKey(n)

def get_vars(f):
    r = []
    def collect(f):
      if z3.is_const(f): 
          if f.decl().kind() == z3.Z3_OP_UNINTERPRETED and not askey(f) in r:
            r.append(askey(f))
      else:
          for c in f.children():
              collect(c)
    collect(f)
    return r

def get_select_exprs(f):
    r = []
    def collect(f):
        if not z3.is_app(f):
            return
        if f.decl().kind() == z3.Z3_OP_SELECT and not askey(f) in r:
            r.append(askey(f))
        if z3.is_const(f): 
            return
        else:
            for c in f.children():
                collect(c)
    collect(f)
    return r

def get_select_exprs_in_goal(goal):
    vars = []
    for expr in goal:
        for var in get_select_exprs(expr):
            if var not in vars:
                vars.append(var)
    return vars

def get_vars_in_goal(goal):
    vars = []
    for expr in goal:
        for var in get_vars(expr):
            if var not in vars:
                vars.append(var)
    return vars

def fmt_to_int(fmt):
    if fmt == 'NHWC' or fmt == 'NDHWC':
        return 0
    elif fmt == 'NCHW' or fmt == 'NCDHW':
        return 1
    elif fmt == 'NCHW_VECT_C':
        return 2
    elif fmt == 'NHWC_VECT_W':
        return 3
    elif fmt == 'HWNC':
        return 4
    elif fmt == 'HWCN':
        return 5
    else:
        assert(0)

def int_to_fmt(val):
    if val == 0:
        return ['NHWC', 'NDHWC']
    elif val == 1:
        return ['NCHW', 'NCDHW']
    elif val == 2:
        return ['NCHW_VECT_C']
    elif val == 3:
        return ['NHWC_VECT_W']
    elif val == 4:
        return ['HWNC']
    elif val == 5:
        return ['HWCN']


def arg_identifier(arg_name):
    return 'ARGNAME_' + arg_name + '_NAMEEND'