nums = '0123456789'
alphabet = 'abcdefghijklmnopqrstuvwxyz'

import numpy as np
import plotly.graph_objects as go

class BinaryFunc:
    def __init__(self, symb, fun):
        self.symb = symb
        self.fun = fun

class UnaryFunc:
    def __init__(self, symb, fun):
        self.symb = symb
        self.fun = fun

unaryfuncs = {
    'ln': lambda x:np.log(x),
    'log': lambda x:np.log10(x),
    'sin': lambda x:np.sin(x),
    'cos': lambda x:np.cos(x),
    'tan': lambda x:np.tan(x),
    'exp': lambda x:np.exp(x),
    'sqrt': lambda x:np.sqrt(x),
    'cbrt': lambda x:np.cbrt(x),
}

def find_dict_index(d, item):
    for i,k in enumerate(d.keys()):
        if k==item:
            return i

def addition_check(string, index):
    left = string[:index]
    right = string[index+1:]
    leftbrackets = rightbrackets = 0
    for innerchar in left[::-1]:
        if innerchar == ')':
            leftbrackets += 1
        elif innerchar == '(':
            rightbrackets += 1
        if leftbrackets < rightbrackets:
            raise AssertionError()

    leftbrackets = rightbrackets = 0
    for innerchar in right:
        if innerchar == ')':
            leftbrackets += 1
        elif innerchar == '(':
            rightbrackets += 1
        if leftbrackets > rightbrackets:
            raise AssertionError()

    return left, right

def subt_check(string, index):
    left = string[:index]
    right = string[index+1:]
    leftbrackets = rightbrackets = 0
    for innerchar in left[::-1]:
        if innerchar == ')':
            leftbrackets += 1
        elif innerchar == '(':
            rightbrackets += 1
        if leftbrackets < rightbrackets:
            raise AssertionError()

    leftbrackets = rightbrackets = 0
    for innerchar in right:
        if innerchar == ')':
            leftbrackets += 1
        elif innerchar == '(':
            rightbrackets += 1
        if leftbrackets > rightbrackets:
            raise AssertionError()
        if innerchar in '+-' and leftbrackets == rightbrackets:
            raise AssertionError()
        
    return left, right
        
def mult_check(string, index):
    left = string[:index]
    right = string[index+1:]
    
    leftbrackets = rightbrackets = 0
    for innerchar in left[::-1]:
        if innerchar == ')':
            leftbrackets += 1
        elif innerchar == '(':
            rightbrackets += 1
        if leftbrackets < rightbrackets:
            raise AssertionError()
        if innerchar in '+-' and leftbrackets == rightbrackets:
            raise AssertionError()
        
    leftbrackets = rightbrackets = 0
    for innerchar in right:
        if innerchar == ')':
            leftbrackets += 1
        elif innerchar == '(':
            rightbrackets += 1
        if leftbrackets > rightbrackets:
            raise AssertionError()
        if innerchar in '+-*/' and leftbrackets == rightbrackets:
            raise AssertionError()
        

    return left, right
        
def div_check(string, index):
    left = string[:index]
    right = string[index+1:]
    
    leftbrackets = rightbrackets = 0
    for innerchar in left[::-1]:
        if innerchar == ')':
            leftbrackets += 1
        elif innerchar == '(':
            rightbrackets += 1
        if rightbrackets > leftbrackets:
            raise AssertionError()
        if innerchar in '+-' and leftbrackets == rightbrackets:
            raise AssertionError()
        
    leftbrackets = rightbrackets = 0
    for innerchar in right:
        if innerchar == ')':
            leftbrackets += 1
        elif innerchar == '(':
            rightbrackets += 1
        if leftbrackets > rightbrackets:
            raise AssertionError()
        if innerchar in '+-*/' and leftbrackets == rightbrackets:
            raise AssertionError()
        
        
    return left, right

def exp_check(string, index):
    left = string[:index]
    right = string[index+1:]
    
    leftbrackets = rightbrackets = 0
    for innerchar in left[::-1]:
        if innerchar == ')':
            leftbrackets += 1
        elif innerchar == '(':
            rightbrackets += 1
        if rightbrackets > leftbrackets:
            raise AssertionError()
        if innerchar in '+-*/' and leftbrackets == rightbrackets:
            raise AssertionError()
        
    leftbrackets = rightbrackets = 0
    for innerchar in right:
        if innerchar == ')':
            leftbrackets += 1
        elif innerchar == '(':
            rightbrackets += 1
        if leftbrackets > rightbrackets:
            raise AssertionError()
        if innerchar in '+-*/^' and leftbrackets == rightbrackets:
            raise AssertionError()
        
    return left, right


def unaryfun_with_func(string, index):
    if index != 0 and string[index-1] in alphabet:
        raise AssertionError()
    # first check if unaryfunc is valid
    string = string[index:]
    func_string = ""
    right_index = 0

    for char in string:
        if char=='(':
            break
        func_string += char
        right_index += 1

    func_content = string[right_index:]

    leftbrackets = rightbrackets = 0
    if func_string != 'logbase':
        for i,char in enumerate(func_content):
            if char == ')':
                leftbrackets += 1
            elif char == '(':
                rightbrackets += 1
            if rightbrackets == leftbrackets:
                if i != len(func_content)-1:
                    raise AssertionError()
    else:
        logbase = ""
        for i,char in enumerate(func_content):
            if char == ')':
                leftbrackets += 1
            elif char == '(':
                rightbrackets += 1
            if leftbrackets == rightbrackets:
                func_content = func_content[i+1:]
                logbase += char
                break
            logbase += char
        for i,char in enumerate(func_content):
            if char == ')':
                leftbrackets += 1
            elif char == '(':
                rightbrackets += 1
            if leftbrackets == rightbrackets:
                if i != len(func_content)-1:
                    raise AssertionError()

        return BinaryFunc('logbase', lambda x,y: np.emath.logn(x, y)), [logbase, func_content]


            
    if func_string[:3] == 'log' and func_string[-1] in nums:
        
        # custom base
        #func = BinaryFunc(f'log{func_string[3:]}', lambda x,y:np.emath.logn(int(func_string[3:]), x))
        func = UnaryFunc(f'log{func_string[3:]}', lambda x:np.emath.logn(int(func_string[3:]), x))
    else:
        func = UnaryFunc(func_string, unaryfuncs[func_string])

    return func, func_content

def find_outer_func(string): # returns UnaryFunc or BinaryFunc

    for i,char in enumerate(string):
        if char == '+':
            try:
                left, right = addition_check(string, i)
                return BinaryFunc('+', lambda x,y:x+y), [left, right]
            except:
                continue
        elif char == '-':
            try:
                left, right = subt_check(string, i)
                return BinaryFunc('-', lambda x,y:x-y), [left, right]
            except:
                continue
        elif char == '*':
            try:
                left, right = mult_check(string, i)
                return BinaryFunc('*', lambda x,y:x*y), [left, right]
            except:
                continue
        elif char == '/':
            try:
                left, right = div_check(string, i)
                return BinaryFunc('/', lambda x,y:x/y), [left, right]
            except:
                continue
        elif char == '^':
            try:
                left, right = exp_check(string, i)
                return BinaryFunc('^', lambda x,y:x**y), [left, right]
            except:
                continue
        
        #elif char in unaryfun_startletters and string[i+1] in alphabet:
        else:
            try:
                return unaryfun_with_func(string, i)
            except:
                continue

def remove_brackets(string):
    if string[0]=='(' and string[-1]==')':
        within = string[1:-1]
        leftbrackets = rightbrackets = 0 
        for char in within:
            if leftbrackets > rightbrackets:
                return string
            if char == ')':
                leftbrackets += 1
            elif char == '(':
                rightbrackets += 1
        return within
    return string

def fc(params, inp):
    args = ",".join(sorted(params))

    return eval(f'lambda {args}:{inp}')

def combine_coefs(*ds):
    r = {}
    for d in ds:
        for k,v in d.items():
            if k in r.keys():
                r[k] += v
            else:
                r[k] = v
    return r

def plot(*funcs, xrange, yrange=None):
    fig = go.Figure()
    if all(f.arity == 1 for f in funcs):
        for f in funcs:
            g = f.get_lambda()
            fig.add_trace(go.Scatter(x=xrange, y=[g(k) for k in xrange]))
        fig.update_layout(xaxis_rangeslider_visible=False, dragmode='pan')
        fig.show(config={'scrollZoom': True, 'displayModeBar': True})
    else:
        assert all(f.arity == 2 for f in funcs) and yrange is not None, AssertionError()
        X, Y = np.meshgrid(xrange, yrange)
        for f in funcs:
            g = f.get_lambda()
            Z = np.array([[g(x_, y_) for x_, y_ in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])
            fig.add_trace(go.Surface(z=Z, x=X, y=Y, showscale=False))
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            dragmode='orbit'
        )
        fig.show(config={'scrollZoom': True, 'displayModeBar': True})

def swap_brackets(string):
    res = ""
    for char in string:
        if char ==')':
            res += '('
        elif char =='(':
            res += ')'
        else:
            res += char
    return res

def tb_removed_on_zero(string: str):
    # string: everything after 0*

    lb=rb=0
    for j,c in enumerate(string):
        if c in '+-' and lb>=rb:

            return string[:j]
        if c==')' and lb>=rb:

            return string[:j]
        if c==')':
            lb+=1
        elif c=='(':
            rb+=1

    return string
        
def tb_reverse(string: str):
    # string: everything before *0


    lb=rb=0
    len_ = len(string)
    for j in range(len_):
        i = len_ - 1 - j
        c = string[i]
        if c in '+-' and lb<=rb:

            return string[i:]
        if c=='(' and lb<=rb:

            return string[i+1:]
        if c==')':
            lb+=1
        elif c=='(':
            rb+=1

    return string

def simplify_str(string: str):

    if not string:
        return '0'
    if len(string)==1:
        return string
    string_orig = string[:]
    
    if string[0]=='0' and string[1]=='*':
        tbr_right = tb_removed_on_zero(string[2:])
        string = string.replace('0*' + tbr_right, '')

    if string[-1]=='0' and string[-2]=='*':
        tbr_left = tb_reverse(string[:-2])

        string = string.replace(tbr_left + '*0', '')

    for i,char in enumerate(string[1:-1], 1):
        if char=='0':
            if string[i-1]=='*':
                if string[i+1]=='*':
                    tbr_left = tb_reverse(string[:i-1])
                    tbr_right = tb_removed_on_zero(string[i+2:])

                    string = string.replace(tbr_left + '*0*' + tbr_right, '')

                    break
                else:
                    tbr_left = tb_reverse(string[:i-1])

                    string = string.replace(tbr_left + '*0', '')
   
                    break
            elif string[i+1]=='*':
                tbr_right = tb_removed_on_zero(string[i+2:])
    
                string = string.replace('0*' + tbr_right, '')

                break

    # remove excess + and -
    res_new = ""
    while res_new != string:
        res_new = string
        string = string.replace('++', '+')
        string = string.replace('--', '+')
        string = string.replace('+-', '-')
        string = string.replace('-+', '-')
        string = string.replace('+)', ')')
        string = string.replace('(+', '(')
        if not string or string in '+-*/^':
            return '0'
        if string[-1] in '+-':
            string = string[:-1]
        if string[0]=='+':
            string = string[1:]

    # remove *1's
    #string = string.replace('1*', '') # with condition
    #string = string.replace('*1', '') # with condition
    
    # remove + or - 0
    #string = string.replace('+0', '') # with condition
    #string = string.replace('-0', '') # with condition
    #string = string.replace('0+', '') # with condition
    #string = string.replace('0-', '-') # with condition
    
    if string_orig == string:
        return string
    return simplify_str(string)

def cte_func(c, vars=None):
    if vars is None:
        return ElementaryFunc(str(c), c, None, ['x'])
    return ElementaryFunc(str(c), c, None, vars)

class ElementaryFunc:
    def __init__(self, string, outside, inside, variables):
        """
        outside: binaryfunc OR unaryfunc OR float/int OR callable
        inside:
        if outside is binary: [elfunc, elfunc]
        if outside is unary: elfunc
        if outside is float/int or callable: none
        """
        self.string = string
        self.outside = outside
        self.inside = inside
        self.variables = sorted(variables)
        self.arity = len(self.variables)

    def add_param(self, param: str):
        self.variables.append(param)
        self.variables = sorted(self.variables)

    def into_constant(self, param: str, value: float | int, replace=True):
        res = ""
        for char in self.string:
            if char==param:
                res += str(value)
            else:
                res += char
        if replace:

            self = make_function(res, simplify=True)
        else:
            return make_function(res, simplify=True)

    def df(self, param=None):

        if param is None:
            assert self.arity == 1, AssertionError()
            param = self.variables[0]
        else:
            assert param in self.variables, AssertionError()

        if isinstance(self.outside, BinaryFunc):

            if self.outside.symb == '+':
                return self.inside[0].df(param) + self.inside[1].df(param)
            elif self.outside.symb == '-':
                return self.inside[0].df(param) - self.inside[1].df(param)
            elif self.outside.symb == '*':
                return self.inside[0].df(param) * self.inside[1] + self.inside[0] * self.inside[1].df(param)
            elif self.outside.symb == '/':
                return (self.inside[0].df(param) * self.inside[1] - self.inside[0] * self.inside[1].df(param)) / self.inside[1] ** 2
            elif self.outside.symb == '^':
                return self * (self.inside[1] * self.inside[0].df(param) / self.inside[0] + self.inside[1].df(param) * ElementaryFunc(
                    f'ln({self.inside[0].string})', UnaryFunc('ln', unaryfuncs['ln']), self.inside[0], self.variables
                ))
            elif self.outside.symb == 'logbase':
                p1 = (self.inside[0].df(param) / self.inside[0]) * ElementaryFunc(f'ln({self.inside[1].string})', UnaryFunc('ln', unaryfuncs['ln']), self.inside[1], self.variables)
                p2 = (self.inside[1].df(param) / self.inside[1]) * ElementaryFunc(f'ln({self.inside[0].string})', UnaryFunc('ln', unaryfuncs['ln']), self.inside[0], self.variables)
                b = ElementaryFunc(f'ln({self.inside[1].string})', UnaryFunc('ln', unaryfuncs['ln']), self.inside[1], self.variables) ** 2
                return (p2 - p1) / b
            

        elif isinstance(self.outside, UnaryFunc):
            
            if self.outside.symb == 'sin':
                return ElementaryFunc(f'cos({self.inside.string})', UnaryFunc('cos', unaryfuncs['cos']), self.inside, self.variables) * self.inside.df(param)
            elif self.outside.symb == 'cos':
                return -ElementaryFunc(f'sin({self.inside.string})', UnaryFunc('cos', unaryfuncs['cos']), self.inside, self.variables) * self.inside.df(param)
            elif self.outside.symb == 'tan':
                return cte_func(1, self.variables) / ElementaryFunc(f'cos({self.inside.string})', UnaryFunc('cos', unaryfuncs['cos']), self.inside, self.variables) ** 2 * (
                    self.inside.df(param)
                )
            elif self.outside.symb == 'exp':
                return self * self.inside.df(param)
            elif self.outside.symb == 'sqrt':
                return cte_func(1, self.variables) / (cte_func(2, self.variables) * self) * self.inside.df(param)
            elif self.outside.symb == 'cbrt':
                f_sq = self.inside ** 2
                cbrt_f_sq = ElementaryFunc(f'cbrt({f_sq.string})', UnaryFunc('cbrt', unaryfuncs['cbrt']), f_sq, self.variables)
                return cte_func(1, self.variables) / (cte_func(3, self.variables) * cbrt_f_sq)
            elif self.outside.symb == 'ln':
                return self.inside.df(param) / self.inside
            elif self.outside.symb == 'log':
                return self.inside.df(param) / (self.inside * ElementaryFunc(f'ln(10)', 10, None, self.variables))

        elif callable(self.outside):
            if param == self.variables[self.outside(*range(self.arity))]:
                return cte_func(1, self.variables)
            return cte_func(0, self.variables)
        return cte_func(0, self.variables)

    def change_vars(self, *args):
        if len(args)==1:
            # dict
            d = args[0]
            assert len(list(d.keys()))==self.arity, AssertionError()
            for k,v in d.items():
                # assumes k is in self.variables
                for i,var in enumerate(self.variables):
                    if k==var:
                        self.variables[i] = v
        else:
            # tuple
            assert len(args)==self.arity, AssertionError()
            for i,arg in enumerate(args):
                self.variables[i] = arg

    def __str__(self):
        start = ""
        for el in sorted(self.variables):
            start += el + ','
        start = start[:-1]
        return f'({start}) => {self.string}'

    def eval(self, *vals):


        if isinstance(self.outside, BinaryFunc):
            # inside is [elfunc, elfunc]
            left = self.inside[0].eval(*vals)
            right = self.inside[1].eval(*vals)
            return self.outside.fun(left, right)
        if isinstance(self.outside, UnaryFunc):
            # inside is elfunc
            return self.outside.fun(self.inside.eval(*vals))
        if callable(self.outside):
            return self.outside(*vals)
        # self.outside is constant
        return self.outside
    
    def get_lambda(self):
        if isinstance(self.outside, UnaryFunc):
            insidefunc = self.inside.get_lambda()
            return lambda *args: self.outside.fun(insidefunc(*args))
        if isinstance(self.outside, BinaryFunc):
            leftfunc = self.inside[0].get_lambda()
            rightfunc = self.inside[1].get_lambda()
            return lambda *args: self.outside.fun(leftfunc(*args), rightfunc(*args))
        if callable(self.outside):
            return lambda *args: self.outside(*args)
        # self.outside is constant
        return lambda *args:self.outside
    
    def compose(self, other):

        def create(self_string, var, other_string):
            res_string = ""
            for char in self_string:
                if char == var:
                    res_string += other_string
                else:
                    res_string += char
            return make_function(res_string)
        
        def multi_create(self_string, info, all_indices):
            res_string = ""
            for i,char in enumerate(self_string):
                if char in info.keys():
                    index = find_dict_index(info, char)
                    try:
                        if i in all_indices[index]:
                            res_string += '('+info[char]+')'
                        else:
                            res_string += info[char]
                    except:
                        # all_indices[index] does not exist
                        res_string += info[char]
                else:
                    res_string += char
            return make_function(res_string)
            
        def x_operators(symb, string, var):
            order = { # other.outside.symb: symbols to ()
                '+': '-*/^',
                '-': '-*/^',
                '*': '/^',
                '/': '/^',
                '^': '^'
            }
            res = []
            for i,char in enumerate(string[1:-1], 1):
                if char == var:
                    if string[i-1] in order[symb]:
                        res.append(i)
                    elif string[i+1] in order[symb]:
                        res.append(i)
            if string[0] == var:
                if symb in order.keys():
                    if string[1] in order[symb]:
                        res.append(0)
            if string[-1] == var:
                if symb in order.keys():
    
                    if string[-2] in order[symb]:
                        res.append(len(string)-1)
            return res
        
        if isinstance(other, ElementaryFunc):
            assert self.arity == 1, AssertionError()

            if isinstance(other.outside, (UnaryFunc, float, int)) or callable(other.outside):
                return create(self.string, self.variables[0], other.string)
            
            # else: other.outside is binaryfunc
            # if log => simple case
            if len(other.outside.symb) >= 3:
                if other.outside.symb[:3] == 'log':
                    return create(self.string, self.variables[0], other.string)
            # else, +-*/^
            indices = x_operators(other.outside.symb, self.string, self.variables[0])
            res_string = ""
            for i,char in enumerate(self.string):
                if i in indices:
                    res_string += '('+other.string+')'
                elif char in self.variables:
                    res_string += other.string
                else:
                    res_string += char
            return make_function(res_string)
        
        # else: other is vectorfunc
        assert self.arity == other.vectordim, AssertionError()
        # f: R^n ->  R
        # g: R^p -> R^n
        # fog: R^p -> R
        info = {}
        all_vars_indeces = [] # list of lists
        for i,elfunc in enumerate(other.elfuncs):
            current_var = self.variables[i]
            info[current_var] = elfunc.string
            if isinstance(elfunc.outside, (UnaryFunc, float, int)) or callable(elfunc.outside):
                all_vars_indeces.append([])
                continue
                
            # elfunc.outside is binary
            elif len(elfunc.outside.symb) >= 3:
                if other.outside.symb[:3] == 'log':
                    all_vars_indeces.append([])
                    continue
            else: # elfunc.outside.symb in '+-*/^
                indices_curvar = x_operators(elfunc.outside.symb, self.string, current_var)

                all_vars_indeces.append(indices_curvar)

        return multi_create(self.string, info, all_vars_indeces)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = cte_func(other, self.variables)
        if other.string[0] != '-':
            return ElementaryFunc(self.string+'+'+other.string,
                              BinaryFunc('+', lambda x,y:x+y),
                              [self, other],
                              list(set(self.variables).union(set(other.variables))))
        return ElementaryFunc(self.string+'+('+other.string+')',
                              BinaryFunc('+', lambda x,y:x+y),
                              [self, other],
                              list(set(self.variables).union(set(other.variables))))
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = cte_func(other, self.variables)
        if isinstance(other.outside, (UnaryFunc, float, int)) or callable(other.outside):
            if other.string[0] != '-':
                newstring = self.string+'-'+other.string
            else:
                newstring = self.string+'-('+other.string+')'
        elif other.outside.symb in '+-':
            newstring = self.string+'-('+other.string+')'
        else:
            newstring = self.string+'-'+other.string
    
        return ElementaryFunc(newstring,
                              BinaryFunc('-', lambda x,y:x-y),
                              [self, other],
                              list(set(self.variables).union(set(other.variables))))
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = cte_func(other, self.variables)
        if isinstance(other.outside, (UnaryFunc, float, int)) or callable(other.outside):
            if other.string[0] != '-':
                if isinstance(self.outside, (UnaryFunc, float, int)) or callable(self.outside):
                    newstring = self.string+'*'+other.string
                elif self.outside.symb in '+-':
                    newstring = '('+self.string+')*'+other.string
                else:
                    newstring = self.string+'*'+other.string
            else:
                if isinstance(self.outside, (UnaryFunc, float, int)) or callable(self.outside):
                    newstring = self.string+'*('+other.string+')'
                elif self.outside.symb in '+-':
                    newstring = '('+self.string+')*('+other.string+')'
                else:
                    newstring = self.string+'*('+other.string+')'
        elif other.outside.symb in '+-':
            if isinstance(self.outside, (UnaryFunc, float, int)) or callable(self.outside):
                newstring = self.string+'*('+other.string+')'
            elif self.outside.symb in '+-':
                newstring = '('+self.string+')*('+other.string+')'
            else:
                newstring = self.string+'*('+other.string+')'
        else:
            if isinstance(self.outside, (UnaryFunc, float, int)) or callable(self.outside):
                newstring = self.string+'*'+other.string
            elif self.outside.symb in '+-':
                newstring = '('+self.string+')*'+other.string
            else:
                newstring = self.string+'*'+other.string
    
        return ElementaryFunc(newstring,
                              BinaryFunc('*', lambda x,y:x*y),
                              [self, other],
                              list(set(self.variables).union(set(other.variables))))
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = cte_func(other, self.variables)
        if isinstance(other.outside, (UnaryFunc, float, int)) or callable(other.outside):
            if other.string[0] != '-':
                if isinstance(self.outside, (UnaryFunc, float, int)) or callable(self.outside):
                    newstring = self.string+'/'+other.string
                elif self.outside.symb in '+-':
                    newstring = '('+self.string+')/'+other.string
                else:
                    newstring = self.string+'/'+other.string
            else:
                if isinstance(self.outside, (UnaryFunc, float, int)) or callable(self.outside):
                    newstring = self.string+'/('+other.string+')'
                elif self.outside.symb in '+-':
                    newstring = '('+self.string+')/('+other.string+')'
                else:
                    newstring = self.string+'/('+other.string+')'
        elif other.outside.symb in '+-*/':
            if isinstance(self.outside, (UnaryFunc, float, int)) or callable(self.outside):
                newstring = self.string+'/('+other.string+')'
            elif self.outside.symb in '+-':
                newstring = '('+self.string+')/('+other.string+')'
            else:
                newstring = self.string+'/('+other.string+')'
        else:
            if isinstance(self.outside, (UnaryFunc, float, int)) or callable(self.outside):
                newstring = self.string+'/'+other.string
            elif self.outside.symb in '+-':
                newstring = '('+self.string+')/'+other.string
            else:
                newstring = self.string+'/'+other.string

        return ElementaryFunc(newstring,
                              BinaryFunc('/', lambda x,y:x/y),
                              [self, other],
                              list(set(self.variables).union(set(other.variables))))
    
    def __xor__(self, other):
        if isinstance(other, (int, float)):
            other = cte_func(other, self.variables)
        if isinstance(other.outside, (UnaryFunc, float, int)) or callable(other.outside):
            if other.string[0] != '-':
                if isinstance(self.outside, (UnaryFunc, float, int)) or callable(self.outside):
                    newstring = self.string+'^'+other.string
                elif self.outside.symb in '+-*/':
                    newstring = '('+self.string+')^'+other.string
                else:
                    newstring = self.string+'^'+other.string
            else:
                if isinstance(self.outside, (UnaryFunc, float, int)) or callable(self.outside):
                    newstring = self.string+'^('+other.string+')'
                elif self.outside.symb in '+-*/':
                    newstring = '('+self.string+')^('+other.string+')'
                else:
                    newstring = self.string+'^('+other.string+')'
        elif other.outside.symb in '+-*/^':
            if isinstance(self.outside, (UnaryFunc, float, int)) or callable(self.outside):
                newstring = self.string+'^('+other.string+')'
            elif self.outside.symb in '+-*/':
                newstring = '('+self.string+')^('+other.string+')'
            else:
                newstring = self.string+'^('+other.string+')'
        else:
            if isinstance(self.outside, (UnaryFunc, float, int)) or callable(self.outside):
                newstring = self.string+'^'+other.string
            elif self.outside.symb in '+-*/':
                newstring = '('+self.string+')^'+other.string
            else:
                newstring = self.string+'^'+other.string

        return ElementaryFunc(newstring,
                              BinaryFunc('^', lambda x,y:x**y),
                              [self, other],
                              list(set(self.variables).union(set(other.variables))))
    
    def __pow__(self, other):
        return self ^ other
    
    def __neg__(self):
 
        if self.string[0]=='-':
            return make_function(self.string[1:], simplify=True)
        if isinstance(self.outside, (float, int, UnaryFunc)) or callable(self.outside):
            return make_function('-' + self.string, simplify=True)
        if self.outside.symb in '+-':
            return make_function('-(' + self.string + ')', simplify=True)
        return make_function('-' + self.string, simplify=True)
    
class VectorFunc:
    def __init__(self, elfuncs: list[ElementaryFunc]):
        self.elfuncs = elfuncs
        self.vectordim = len(elfuncs)
        variables = set()
        for elfunc in elfuncs:
            for var in elfunc.variables:
                variables.add(var)
        self.variables = sorted(list(variables))
        self.arity = len(variables)

    def get_lambda(self):

        funcs = []
        for ef in self.elfuncs:
            l = ef.get_lambda()
            funcs.append(l)

        return lambda *args, **kwargs: tuple(f(*args, **kwargs) for f in funcs)
    
    def eval(self, *vals):
        return tuple(ef.eval(*vals) for ef in self.elfuncs)
            
    def get_string_list(self):
        res = []
        for ef in self.elfuncs:
            res.append(ef.string)
        return res

    def __str__(self):
        start = ""
        for el in sorted(self.variables):
            start += el + ','
        start = start[:-1]
        output = "("
        for ef in self.elfuncs:
            output += ef.string+', ' 
        return f'({start}) => {output[:-2]})'
    
    def __add__(self, other):
        assert self.vectordim == other.vectordim, AssertionError()
        res_elfuncs = []
        for self_elfunc, other_elfunc in zip(self.elfuncs, other.elfuncs):
            new_elfunc = self_elfunc + other_elfunc
            res_elfuncs.append(new_elfunc)
        return VectorFunc(res_elfuncs)
    
    def __sub__(self, other):
        assert self.vectordim == other.vectordim, AssertionError()
        res_elfuncs = []
        for self_elfunc, other_elfunc in zip(self.elfuncs, other.elfuncs):
            new_elfunc = self_elfunc - other_elfunc
            res_elfuncs.append(new_elfunc)
        return VectorFunc(res_elfuncs)
    
    def __mul__(self, other):
        assert self.vectordim == other.vectordim, AssertionError()
        res_elfuncs = []
        for self_elfunc, other_elfunc in zip(self.elfuncs, other.elfuncs):
            new_elfunc = self_elfunc * other_elfunc
            res_elfuncs.append(new_elfunc)
        return VectorFunc(res_elfuncs)
    
    def __truediv__(self, other):
        assert self.vectordim == other.vectordim, AssertionError()
        res_elfuncs = []
        for self_elfunc, other_elfunc in zip(self.elfuncs, other.elfuncs):
            new_elfunc = self_elfunc / other_elfunc
            res_elfuncs.append(new_elfunc)
        return VectorFunc(res_elfuncs)
    
    def __xor__(self, other):
        assert self.vectordim == other.vectordim, AssertionError()
        res_elfuncs = []
        for self_elfunc, other_elfunc in zip(self.elfuncs, other.elfuncs):
            new_elfunc = self_elfunc ^ other_elfunc
            res_elfuncs.append(new_elfunc)
        return VectorFunc(res_elfuncs)
    
    def __pow__(self, other):
        return self ^ other

    def compose(self, other):
        if isinstance(other, ElementaryFunc):
            assert self.arity == 1, AssertionError()
            res_elfuncs = []
            for elfunc in self.elfuncs:
                res_elfuncs.append(elfunc.compose(other))
            return VectorFunc(res_elfuncs)
        
        # other is vectorfunc
        assert self.arity == other.vectordim, AssertionError()
        res = []
        for self_elfunc in self.elfuncs:
            f = self_elfunc.compose(other)
            res.append(f)
        return VectorFunc(res)


def get_variables(string):
    # returns set
    variables = set()
    if len(string) != 1:
        if string[0] in alphabet and string[1] not in alphabet:
            variables.add(string[0])
        if string[-1] in alphabet and string[-2] not in alphabet:
            variables.add(string[-1])
        for i,char in enumerate(string[1:-1], 1):
            if char in alphabet and string[i-1] not in alphabet and string[i+1] not in alphabet:
                variables.add(char)
    else:
        if string in alphabet:
            variables = set(string)

    return variables

def make_function(*strings, simplify=False):

    elfuncs = []
    for string in strings:
        string = remove_brackets(string)
        if simplify:
            string = simplify_str(string)
        
        variables = set()
        for v in [get_variables(s) for s in strings]:
            variables = variables.union(v)
        variables = sorted(list(variables))

        if not variables:
            variables = ['x']
        
        if len(strings)==1:
            return func_constructor(string, variables)
        
        elfunc = func_constructor(string, variables)
        elfuncs.append(elfunc)

    return VectorFunc(elfuncs)

def func_constructor(string, variables):
    #string = remove_brackets(string)


    string = remove_brackets(string)
    if all(s in nums or s=='.' for s in string):
        return ElementaryFunc(string, outside=float(string), inside=None, variables=variables)
    elif string in ['e', 'pi']:
        return ElementaryFunc(string, outside=eval(f'lambda {",".join(variables)}:np.{string}'), inside=None, variables=variables)
    elif string in variables:
        return ElementaryFunc(string, outside=fc(variables, string), inside=None, variables=variables)
    elif string[0] == '-':
        insidefunc = func_constructor(string[1:], variables)
        return ElementaryFunc(string, UnaryFunc('-', lambda x:-x), insidefunc, variables)
    
    else:
        outsidefunc, content = find_outer_func(string)
        if isinstance(outsidefunc, BinaryFunc):
            insidefuncs = [
                func_constructor(content[0], variables),
                func_constructor(content[1], variables)
            ]
            return ElementaryFunc(string, outsidefunc, insidefuncs, variables)
        else:

            insidefunc = func_constructor(content, variables)
            return ElementaryFunc(string, outsidefunc, insidefunc, variables)
        

