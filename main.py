# MiniCalc: Single-file calculus tool with a terminal UI (NO imports).
# Features:
# - Parse & evaluate expressions (variables + common functions)
# - Symbolic derivatives (single & multivariable), gradient, Jacobian, Hessian
# - Divergence & curl (3D vector fields)
# - Numerical derivatives + numerical definite integrals (Trapezoid / Simpson / Adaptive Simpson)
# - Symbolic indefinite integrals (limited built-in rules)
# - Double integrals over rectangles (nested Simpson)
#
# How to run:
#   1) Save as: minicalc.py
#   2) Run:     python3 minicalc.py
#
# Expression syntax:
#   - Operators: +  -  *  /  ^   (power)
#   - Functions: sin(x), cos(x), tan(x), exp(x), ln(x), sqrt(x), abs(x)
#   - Constants: pi, e
#   - Use explicit multiplication: 2*x (not 2x)
#
# Notes:
# - Numeric sin/cos/exp/ln are implemented via built-in logic (series/range reduction).
# - Symbolic simplification is intentionally lightweight (educational, not a full CAS).

# ---------------------------- Numeric core (no imports) ----------------------------

PI = 3.14159265358979323846264338327950288419716939937510
TAU = 2.0 * PI
LN2 = 0.69314718055994530941723212145817656807550013436026
E  = 2.71828182845904523536028747135266249775724709369995

def _abs(x):
    return -x if x < 0 else x

def _wrap_angle(x):
    # Reduce x to [-PI, PI] without imports.
    if x == 0.0:
        return 0.0
    k = int(x / TAU)  # trunc toward 0
    if x < 0.0 and x != k * TAU:
        k -= 1  # emulate floor for negatives
    r = x - k * TAU   # now in [0, 2pi)
    if r > PI:
        r -= TAU
    return r

def n_sin(x):
    x = _wrap_angle(x)
    term = x
    s = x
    x2 = x * x
    n = 1
    while n <= 12:
        term *= -x2 / ((2*n) * (2*n + 1))
        s += term
        n += 1
    return s

def n_cos(x):
    x = _wrap_angle(x)
    term = 1.0
    c = 1.0
    x2 = x * x
    n = 1
    while n <= 12:
        term *= -x2 / ((2*n - 1) * (2*n))
        c += term
        n += 1
    return c

def n_tan(x):
    c = n_cos(x)
    if c == 0.0:
        raise ValueError("tan undefined (cos(x)=0)")
    return n_sin(x) / c

def n_exp(x):
    # Range reduction: x = k*ln2 + r, exp(x)=2^k * exp(r), r in ~[-ln2/2, ln2/2]
    if x == 0.0:
        return 1.0
    k = int(x / LN2)
    if x < 0.0 and x != k * LN2:
        k -= 1
    r = x - k * LN2
    if r > 0.5 * LN2:
        r -= LN2
        k += 1
    elif r < -0.5 * LN2:
        r += LN2
        k -= 1

    term = 1.0
    s = 1.0
    n = 1
    while n <= 30:
        term *= r / n
        s += term
        n += 1
    return (2.0 ** k) * s

def n_ln(x):
    if x <= 0.0:
        raise ValueError("ln domain error (x must be > 0)")
    # Reduce to m in [0.75, 1.5]
    n = 0
    m = x
    while m > 1.5:
        m *= 0.5
        n += 1
    while m < 0.75:
        m *= 2.0
        n -= 1

    # atanh series: ln(m)=2*(t + t^3/3 + t^5/5 + ...), t=(m-1)/(m+1)
    t = (m - 1.0) / (m + 1.0)
    t2 = t * t
    term = t
    s = 0.0
    k = 1
    while k <= 79:
        s += term / k
        term *= t2
        k += 2
    return 2.0 * s + n * LN2

def n_sqrt(x):
    if x < 0.0:
        raise ValueError("sqrt domain error (x must be >= 0)")
    if x == 0.0:
        return 0.0
    g = x if x < 1.0 else x * 0.5
    i = 0
    while i < 25:
        g = 0.5 * (g + x / g)
        i += 1
    return g

def n_abs(x):
    return _abs(x)

def n_pow(a, b):
    return a ** b

# ---------------------------- AST & Parsing ----------------------------

# Node kinds:
# - ('num', value)
# - ('var', name)
# - ('op', op, left, right)      for binary ops
# - ('uop', op, child)           for unary ops ('neg' only)
# - ('func', name, [args...])

def Num(v): return ('num', float(v))
def Var(n): return ('var', n)
def Op(op, a, b): return ('op', op, a, b)
def UOp(op, a): return ('uop', op, a)
def Func(name, args): return ('func', name, args)

_PRECEDENCE = {'+':1, '-':1, '*':2, '/':2, '^':3}
_RIGHT_ASSOC = {'^': True}

_FUNCTIONS = {
    'sin':1, 'cos':1, 'tan':1,
    'exp':1, 'ln':1, 'sqrt':1, 'abs':1
}

_CONSTANTS = {'pi': PI, 'e': E}

def tokenize(s):
    tokens = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch in ' \t\r\n':
            i += 1
            continue

        # number (supports simple scientific notation like 1e-3)
        if ch.isdigit() or ch == '.':
            j = i
            dot = 0
            while j < n and (s[j].isdigit() or s[j] == '.'):
                if s[j] == '.':
                    dot += 1
                    if dot > 1:
                        break
                j += 1
            # exponent part
            if j < n and (s[j] == 'e' or s[j] == 'E'):
                k = j + 1
                if k < n and (s[k] == '+' or s[k] == '-'):
                    k += 1
                if k < n and s[k].isdigit():
                    k2 = k
                    while k2 < n and s[k2].isdigit():
                        k2 += 1
                    j = k2
            tokens.append(('num', s[i:j]))
            i = j
            continue

        # identifier
        if ch.isalpha() or ch == '_':
            j = i
            while j < n and (s[j].isalnum() or s[j] == '_'):
                j += 1
            tokens.append(('id', s[i:j]))
            i = j
            continue

        # symbols
        if ch in '+-*/^(),':
            tokens.append(('sym', ch))
            i += 1
            continue

        raise ValueError("Unexpected character: " + ch)
    return tokens

def parse(expr):
    tokens = tokenize(expr)
    output = []
    ops = []
    prev = None
    i = 0

    def push_op(op):
        while ops:
            top = ops[-1]
            if top[0] == 'sym' and top[1] in _PRECEDENCE:
                p1 = _PRECEDENCE[op]
                p2 = _PRECEDENCE[top[1]]
                if (p2 > p1) or (p2 == p1 and not _RIGHT_ASSOC.get(op, False)):
                    output.append(ops.pop())
                    continue
            break
        ops.append(('sym', op))

    while i < len(tokens):
        ttype, tval = tokens[i]

        if ttype == 'num':
            output.append(('num', float(tval)))
            prev = 'atom'

        elif ttype == 'id':
            name = tval
            nxt = tokens[i+1] if i+1 < len(tokens) else None
            if nxt and nxt[0] == 'sym' and nxt[1] == '(' and name in _FUNCTIONS:
                ops.append(('func', name))
                ops.append(('sym', '('))
                prev = None
                i += 1  # skip '(' token (we already pushed it)
            else:
                if name in _CONSTANTS:
                    output.append(('num', float(_CONSTANTS[name])))
                else:
                    output.append(('var', name))
                prev = 'atom'

        elif ttype == 'sym':
            if tval == '(':
                ops.append(('sym', '('))
                prev = None

            elif tval == ')':
                while ops and not (ops[-1][0] == 'sym' and ops[-1][1] == '('):
                    output.append(ops.pop())
                if not ops:
                    raise ValueError("Mismatched parentheses")
                ops.pop()  # pop '('
                if ops and ops[-1][0] == 'func':
                    output.append(ops.pop())
                prev = 'atom'

            elif tval == ',':
                while ops and not (ops[-1][0] == 'sym' and ops[-1][1] == '('):
                    output.append(ops.pop())
                if not ops:
                    raise ValueError("Misplaced comma")
                prev = None

            elif tval in '+-*/^':
                if tval == '-' and (prev is None or prev == 'op'):
                    ops.append(('uop', 'neg'))
                else:
                    push_op(tval)
                    prev = 'op'
            else:
                raise ValueError("Unknown symbol: " + tval)
        else:
            raise ValueError("Bad token")

        i += 1

    while ops:
        top = ops.pop()
        if top[0] == 'sym' and top[1] in '()':
            raise ValueError("Mismatched parentheses")
        output.append(top)

    # Build AST from RPN
    stack = []
    for tok in output:
        if tok[0] == 'num':
            stack.append(Num(tok[1]))
        elif tok[0] == 'var':
            stack.append(Var(tok[1]))
        elif tok[0] == 'uop' and tok[1] == 'neg':
            if not stack:
                raise ValueError("Unary '-' missing operand")
            a = stack.pop()
            stack.append(UOp('neg', a))
        elif tok[0] == 'sym' and tok[1] in _PRECEDENCE:
            if len(stack) < 2:
                raise ValueError("Binary operator missing operands: " + tok[1])
            b = stack.pop()
            a = stack.pop()
            stack.append(Op(tok[1], a, b))
        elif tok[0] == 'func':
            fname = tok[1]
            if not stack:
                raise ValueError("Function missing args: " + fname)
            a = stack.pop()
            stack.append(Func(fname, [a]))
        else:
            raise ValueError("Bad RPN token: " + str(tok))

    if len(stack) != 1:
        raise ValueError("Invalid expression")
    return stack[0]

# ---------------------------- Evaluation ----------------------------

def eval_ast(node, env):
    k = node[0]
    if k == 'num':
        return node[1]
    if k == 'var':
        name = node[1]
        if name in env:
            return float(env[name])
        raise ValueError("Missing variable value: " + name)
    if k == 'uop':
        op = node[1]
        a = eval_ast(node[2], env)
        if op == 'neg':
            return -a
        raise ValueError("Unknown unary op: " + op)
    if k == 'op':
        op, aN, bN = node[1], node[2], node[3]
        a = eval_ast(aN, env)
        b = eval_ast(bN, env)
        if op == '+': return a + b
        if op == '-': return a - b
        if op == '*': return a * b
        if op == '/':
            if b == 0.0: raise ValueError("Division by zero")
            return a / b
        if op == '^': return n_pow(a, b)
        raise ValueError("Unknown op: " + op)
    if k == 'func':
        name = node[1]
        x = eval_ast(node[2][0], env)
        if name == 'sin': return n_sin(x)
        if name == 'cos': return n_cos(x)
        if name == 'tan': return n_tan(x)
        if name == 'exp': return n_exp(x)
        if name == 'ln':  return n_ln(x)
        if name == 'sqrt':return n_sqrt(x)
        if name == 'abs': return n_abs(x)
        raise ValueError("Unknown function: " + name)
    raise ValueError("Unknown node kind: " + k)

# ---------------------------- Pretty printing ----------------------------

def _prec(node):
    if node[0] == 'op':
        return _PRECEDENCE.get(node[1], 99)
    if node[0] == 'uop':
        return 4
    return 99

def to_str(node):
    k = node[0]
    if k == 'num':
        v = node[1]
        if v == int(v):
            return str(int(v))
        s = str(v)
        if 'e' not in s and 'E' not in s and '.' in s:
            s = s.rstrip('0').rstrip('.')
        return s
    if k == 'var':
        return node[1]
    if k == 'uop':
        a = node[2]
        s = to_str(a)
        if a[0] == 'op':
            s = '(' + s + ')'
        return '-' + s
    if k == 'op':
        op, a, b = node[1], node[2], node[3]
        pa = _prec(a)
        pb = _prec(b)
        p = _PRECEDENCE.get(op, 0)
        sa = to_str(a)
        sb = to_str(b)
        if a[0] == 'op' and pa < p:
            sa = '(' + sa + ')'
        if b[0] == 'op':
            need = (pb < p) or (op == '^' and pb == p)
            if need:
                sb = '(' + sb + ')'
        return sa + op + sb
    if k == 'func':
        return node[1] + '(' + to_str(node[2][0]) + ')'
    return str(node)

# ---------------------------- Simplification ----------------------------

def _is_zero(n): return n[0] == 'num' and n[1] == 0.0
def _is_one(n):  return n[0] == 'num' and n[1] == 1.0

def rewrite_tan(node):
    # Replace tan(u) with sin(u)/cos(u) to keep things simple/supported.
    k = node[0]
    if k in ('num','var'):
        return node
    if k == 'uop':
        return UOp(node[1], rewrite_tan(node[2]))
    if k == 'op':
        return Op(node[1], rewrite_tan(node[2]), rewrite_tan(node[3]))
    if k == 'func':
        name = node[1]
        u = rewrite_tan(node[2][0])
        if name == 'tan':
            return Op('/', Func('sin', [u]), Func('cos', [u]))
        return Func(name, [u])
    return node

def simplify(node):
    k = node[0]
    if k in ('num', 'var'):
        return node

    if k == 'uop':
        a = simplify(node[2])
        if node[1] == 'neg':
            if a[0] == 'num':
                return Num(-a[1])
            if a[0] == 'uop' and a[1] == 'neg':
                return simplify(a[2])
        return UOp(node[1], a)

    if k == 'func':
        a = simplify(node[2][0])
        if a[0] == 'num':
            try:
                return Num(eval_ast(Func(node[1], [a]), {}))
            except:
                pass
        return Func(node[1], [a])

    if k == 'op':
        op = node[1]
        a = simplify(node[2])
        b = simplify(node[3])

        # constant fold
        if a[0] == 'num' and b[0] == 'num':
            try:
                return Num(eval_ast(Op(op, a, b), {}))
            except:
                pass

        # algebraic identities
        if op == '+':
            if _is_zero(a): return b
            if _is_zero(b): return a
        elif op == '-':
            if _is_zero(b): return a
            if _is_zero(a): return UOp('neg', b)
        elif op == '*':
            if _is_zero(a) or _is_zero(b): return Num(0.0)
            if _is_one(a): return b
            if _is_one(b): return a
        elif op == '/':
            if _is_zero(a): return Num(0.0)
            if _is_one(b): return a
        elif op == '^':
            if _is_zero(b): return Num(1.0)
            if _is_one(b): return a
            if _is_zero(a): return Num(0.0)

        # extra sign/constant simplifications (still lightweight)
        if op == '-' and b[0] == 'uop' and b[1] == 'neg':
            return simplify(Op('+', a, b[2]))
        if op == '+' and b[0] == 'uop' and b[1] == 'neg':
            return simplify(Op('-', a, b[2]))

        # (k*u)/m -> (k/m)*u
        if op == '/' and a[0] == 'op' and a[1] == '*' and b[0] == 'num':
            L, R = a[2], a[3]
            if L[0] == 'num' and b[1] != 0.0:
                return simplify(Op('*', Num(L[1] / b[1]), R))
            if R[0] == 'num' and b[1] != 0.0:
                return simplify(Op('*', Num(R[1] / b[1]), L))

        # u/(k*v) -> (1/k) * (u/v)
        if op == '/' and b[0] == 'op' and b[1] == '*':
            L, R = b[2], b[3]
            if L[0] == 'num' and L[1] != 0.0:
                return simplify(Op('*', Num(1.0 / L[1]), Op('/', a, R)))
            if R[0] == 'num' and R[1] != 0.0:
                return simplify(Op('*', Num(1.0 / R[1]), Op('/', a, L)))

        # k*(u/m) -> (k/m)*u
        if op == '*' and a[0] == 'num' and b[0] == 'op' and b[1] == '/' and b[3][0] == 'num' and b[3][1] != 0.0:
            return simplify(Op('*', Num(a[1] / b[3][1]), b[2]))
        if op == '*' and b[0] == 'num' and a[0] == 'op' and a[1] == '/' and a[3][0] == 'num' and a[3][1] != 0.0:
            return simplify(Op('*', Num(b[1] / a[3][1]), a[2]))

        # (-a)*b or a*(-b)
        if op == '*' and a[0] == 'uop' and a[1] == 'neg':
            return simplify(UOp('neg', Op('*', a[2], b)))
        if op == '*' and b[0] == 'uop' and b[1] == 'neg':
            return simplify(UOp('neg', Op('*', a, b[2])))

        return Op(op, a, b)

    return node

# ---------------------------- Symbolic Differentiation ----------------------------

def d(node, var):
    k = node[0]
    if k == 'num':
        return Num(0.0)
    if k == 'var':
        return Num(1.0) if node[1] == var else Num(0.0)
    if k == 'uop':
        if node[1] == 'neg':
            return UOp('neg', d(node[2], var))
        return Num(0.0)
    if k == 'op':
        op, u, v = node[1], node[2], node[3]
        du = d(u, var)
        dv = d(v, var)
        if op == '+': return Op('+', du, dv)
        if op == '-': return Op('-', du, dv)
        if op == '*': return Op('+', Op('*', du, v), Op('*', u, dv))
        if op == '/':
            num = Op('-', Op('*', du, v), Op('*', u, dv))
            den = Op('^', v, Num(2.0))
            return Op('/', num, den)
        if op == '^':
            if v[0] == 'num':
                c = v[1]
                return Op('*', Op('*', Num(c), Op('^', u, Num(c - 1.0))), du)
            if u[0] == 'num':
                a = u[1]
                return Op('*', Op('*', Op('^', Num(a), v), Func('ln', [Num(a)])), dv)
            term1 = Op('*', dv, Func('ln', [u]))
            term2 = Op('*', v, Op('/', du, u))
            return Op('*', Op('^', u, v), Op('+', term1, term2))
        raise ValueError("Unknown op for differentiation: " + op)
    if k == 'func':
        name = node[1]
        u = node[2][0]
        du = d(u, var)
        if name == 'sin': return Op('*', Func('cos', [u]), du)
        if name == 'cos': return Op('*', UOp('neg', Func('sin', [u])), du)
        if name == 'exp': return Op('*', Func('exp', [u]), du)
        if name == 'ln':  return Op('*', Op('/', Num(1.0), u), du)
        if name == 'sqrt':return Op('*', Op('/', Num(1.0), Op('*', Num(2.0), Func('sqrt', [u]))), du)
        if name == 'abs': return Op('*', Op('/', u, Func('abs', [u])), du)  # undefined at 0
        raise ValueError("Unsupported function for differentiation: " + name)
    raise ValueError("Unknown node kind for differentiation: " + k)

def deriv(node, var):
    dn = d(rewrite_tan(node), var)
    return simplify(dn)

# ---------------------------- Multivariable tools ----------------------------

def gradient(f_node, vars_list):
    return [simplify(deriv(f_node, v)) for v in vars_list]

def jacobian(vec_nodes, vars_list):
    J = []
    for fi in vec_nodes:
        row = []
        for v in vars_list:
            row.append(simplify(deriv(fi, v)))
        J.append(row)
    return J

def hessian(f_node, vars_list):
    H = []
    for vi in vars_list:
        row = []
        d1 = simplify(deriv(f_node, vi))
        for vj in vars_list:
            row.append(simplify(deriv(d1, vj)))
        H.append(row)
    return H

def divergence(F_nodes, vars_list):
    if len(F_nodes) != len(vars_list):
        raise ValueError("Divergence needs same number of components as variables")
    s = Num(0.0)
    i = 0
    while i < len(vars_list):
        s = Op('+', s, deriv(F_nodes[i], vars_list[i]))
        i += 1
    return simplify(s)

def curl_3d(F_nodes, vars_list):
    if len(F_nodes) != 3 or len(vars_list) != 3:
        raise ValueError("Curl here is 3D only: vars=[x,y,z], F=[Fx,Fy,Fz]")
    x,y,z = vars_list[0], vars_list[1], vars_list[2]
    Fx,Fy,Fz = F_nodes[0], F_nodes[1], F_nodes[2]
    c1 = simplify(Op('-', deriv(Fz, y), deriv(Fy, z)))
    c2 = simplify(Op('-', deriv(Fx, z), deriv(Fz, x)))
    c3 = simplify(Op('-', deriv(Fy, x), deriv(Fx, y)))
    return [c1, c2, c3]

# ---------------------------- Symbolic Integration (limited rules) ----------------------------

def depends_on(node, var):
    k = node[0]
    if k == 'var':
        return node[1] == var
    if k == 'num':
        return False
    if k == 'uop':
        return depends_on(node[2], var)
    if k == 'op':
        return depends_on(node[2], var) or depends_on(node[3], var)
    if k == 'func':
        return depends_on(node[2][0], var)
    return True

def _is_linear_in(node, var):
    # Returns (A,B) such that node == A*var + B, with A,B not depending on var.
    n = simplify(node)

    if n[0] == 'var' and n[1] == var:
        return (Num(1.0), Num(0.0))

    if n[0] == 'op' and n[1] == '*':
        L, R = n[2], n[3]
        if L[0] == 'var' and L[1] == var and not depends_on(R, var):
            return (R, Num(0.0))
        if R[0] == 'var' and R[1] == var and not depends_on(L, var):
            return (L, Num(0.0))

    if n[0] == 'op' and n[1] in ('+','-'):
        L, R = n[2], n[3]
        linL = _is_linear_in(L, var)
        if linL and not depends_on(R, var):
            A, B = linL
            if n[1] == '+':
                return (A, simplify(Op('+', B, R)))
            else:
                return (A, simplify(Op('-', B, R)))
        linR = _is_linear_in(R, var)
        if linR and not depends_on(L, var):
            A, B = linR
            if n[1] == '+':
                return (A, simplify(Op('+', L, B)))
            else:
                return (simplify(UOp('neg', A)), simplify(Op('-', L, B)))

    return None

def integrate(node, var):
    """
    Symbolic indefinite integral ∫ node d(var) using a small built-in rule set.
    Raises ValueError if it can't find a rule.
    """
    n = simplify(rewrite_tan(node))

    # ∫ c dx = c*x
    if n[0] == 'num':
        return simplify(Op('*', n, Var(var)))

    # ∫ x dx ; ∫ otherVar dx
    if n[0] == 'var':
        if n[1] == var:
            return simplify(Op('/', Op('^', Var(var), Num(2.0)), Num(2.0)))
        return simplify(Op('*', n, Var(var)))

    # ∫ -u dx
    if n[0] == 'uop' and n[1] == 'neg':
        return simplify(UOp('neg', integrate(n[2], var)))

    # linearity
    if n[0] == 'op' and n[1] in ('+','-'):
        return simplify(Op(n[1], integrate(n[2], var), integrate(n[3], var)))

    # constant multiple
    if n[0] == 'op' and n[1] == '*':
        a, b = n[2], n[3]
        if not depends_on(a, var):
            return simplify(Op('*', a, integrate(b, var)))
        if not depends_on(b, var):
            return simplify(Op('*', b, integrate(a, var)))

    # constant divisor / linear denominator
    if n[0] == 'op' and n[1] == '/':
        a, b = n[2], n[3]
        if not depends_on(b, var):
            return simplify(Op('/', integrate(a, var), b))

        lin = _is_linear_in(b, var)
        if lin and not depends_on(a, var):
            A, B = lin
            return simplify(Op('*', Op('/', a, A), Func('ln', [Func('abs', [b])])))

        if a[0] == 'num' and a[1] == 1.0 and lin:
            A, B = lin
            return simplify(Op('*', Op('/', Num(1.0), A), Func('ln', [Func('abs', [b])])))

        if a[0] == 'num' and a[1] == 1.0 and b[0] == 'var' and b[1] == var:
            return Func('ln', [Func('abs', [Var(var)])])

    # power rule: ∫ x^p dx
    if n[0] == 'op' and n[1] == '^':
        base, expn = n[2], n[3]
        if base[0] == 'var' and base[1] == var and expn[0] == 'num':
            p = expn[1]
            if p == -1.0:
                return Func('ln', [Func('abs', [Var(var)])])
            return simplify(Op('/', Op('^', Var(var), Num(p + 1.0)), Num(p + 1.0)))

    # trig/exp with linear inner
    if n[0] == 'func':
        fname = n[1]
        u = n[2][0]
        lin = _is_linear_in(u, var)
        if lin:
            A, B = lin
            if fname == 'exp':
                return simplify(Op('/', Func('exp', [u]), A))
            if fname == 'sin':
                return simplify(UOp('neg', Op('/', Func('cos', [u]), A)))
            if fname == 'cos':
                return simplify(Op('/', Func('sin', [u]), A))

        if fname == 'exp' and u[0] == 'var' and u[1] == var:
            return Func('exp', [Var(var)])
        if fname == 'sin' and u[0] == 'var' and u[1] == var:
            return UOp('neg', Func('cos', [Var(var)]))
        if fname == 'cos' and u[0] == 'var' and u[1] == var:
            return Func('sin', [Var(var)])

    raise ValueError("No symbolic integration rule for: " + to_str(n))

# ---------------------------- Numerical calculus ----------------------------

def num_derivative(f_node, var, x0, h, env):
    env2 = {}
    for k in env:
        env2[k] = env[k]
    env2[var] = x0 + h
    fp = eval_ast(f_node, env2)
    env2[var] = x0 - h
    fm = eval_ast(f_node, env2)
    return (fp - fm) / (2.0 * h)

def trapz_1d(f_node, var, a, b, n, env):
    if n <= 0:
        raise ValueError("n must be positive")
    h = (b - a) / n
    s = 0.0
    env2 = {}
    for k in env:
        env2[k] = env[k]
    i = 0
    while i <= n:
        x = a + i * h
        env2[var] = x
        fx = eval_ast(f_node, env2)
        if i == 0 or i == n:
            s += 0.5 * fx
        else:
            s += fx
        i += 1
    return s * h

def simpson_1d(f_node, var, a, b, n, env):
    if n <= 0:
        raise ValueError("n must be positive")
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    env2 = {}
    for k in env:
        env2[k] = env[k]
    s = 0.0
    i = 0
    while i <= n:
        x = a + i * h
        env2[var] = x
        fx = eval_ast(f_node, env2)
        if i == 0 or i == n:
            w = 1.0
        elif i % 2 == 1:
            w = 4.0
        else:
            w = 2.0
        s += w * fx
        i += 1
    return s * h / 3.0

def _simpson_raw(f, a, b, fa, fm, fb):
    return (b - a) * (fa + 4.0 * fm + fb) / 6.0

def adaptive_simpson_1d(f_node, var, a, b, eps, env, max_depth):
    env2 = {}
    for k in env:
        env2[k] = env[k]

    def f(x):
        env2[var] = x
        return eval_ast(f_node, env2)

    fa = f(a)
    fb = f(b)
    m = 0.5 * (a + b)
    fm = f(m)
    whole = _simpson_raw(f, a, b, fa, fm, fb)

    def rec(a, b, fa, fm, fb, whole, depth):
        m = 0.5 * (a + b)
        lm = 0.5 * (a + m)
        rm = 0.5 * (m + b)
        flm = f(lm)
        frm = f(rm)
        left = _simpson_raw(f, a, m, fa, flm, fm)
        right = _simpson_raw(f, m, b, fm, frm, fb)
        if depth <= 0:
            return left + right
        if _abs(left + right - whole) <= 15.0 * eps:
            return left + right + (left + right - whole) / 15.0
        return rec(a, m, fa, flm, fm, left, depth - 1) + rec(m, b, fm, frm, fb, right, depth - 1)

    return rec(a, b, fa, fm, fb, whole, max_depth)

def simpson_2d(f_node, varx, a, b, nx, vary, c, d, ny, env):
    if nx % 2 == 1: nx += 1
    if ny % 2 == 1: ny += 1
    hx = (b - a) / nx
    hy = (d - c) / ny
    env2 = {}
    for k in env:
        env2[k] = env[k]

    def inner_integral(x):
        env2[varx] = x
        s = 0.0
        j = 0
        while j <= ny:
            y = c + j * hy
            env2[vary] = y
            fxy = eval_ast(f_node, env2)
            if j == 0 or j == ny:
                w = 1.0
            elif j % 2 == 1:
                w = 4.0
            else:
                w = 2.0
            s += w * fxy
            j += 1
        return s * hy / 3.0

    S = 0.0
    i = 0
    while i <= nx:
        x = a + i * hx
        gx = inner_integral(x)
        if i == 0 or i == nx:
            w = 1.0
        elif i % 2 == 1:
            w = 4.0
        else:
            w = 2.0
        S += w * gx
        i += 1
    return S * hx / 3.0

# ---------------------------- Terminal UI helpers ----------------------------

CSI = "\x1b["
def clr():
    print(CSI + "2J" + CSI + "H", end="")

def color(s, code):
    return CSI + code + "m" + s + CSI + "0m"

def box(title, lines):
    w = len(title) + 4
    i = 0
    while i < len(lines):
        if len(lines[i]) + 4 > w:
            w = len(lines[i]) + 4
        i += 1
    top = "┌" + "─"*(w-2) + "┐"
    midt = "│ " + title + " "*(w-3-len(title)) + "│"
    sep = "├" + "─"*(w-2) + "┤"
    out = [top, midt, sep]
    for ln in lines:
        out.append("│ " + ln + " "*(w-3-len(ln)) + "│")
    out.append("└" + "─"*(w-2) + "┘")
    return "\n".join(out)

def parse_vars_list(s):
    parts = [p.strip() for p in s.split(',') if p.strip() != ""]
    if not parts:
        raise ValueError("No variables provided")
    return parts

def prompt_vars(vars_list):
    env = {}
    for v in vars_list:
        raw = input("  " + v + " = ").strip()
        if raw == "":
            raise ValueError("Missing value for " + v)
        env[v] = float(raw)
    return env

def pretty_vector(V):
    return "[ " + ", ".join([to_str(x) for x in V]) + " ]"

def pretty_matrix(M):
    rows = []
    for r in M:
        rows.append("[ " + ", ".join([to_str(x) for x in r]) + " ]")
    return "[\n  " + ",\n  ".join(rows) + "\n]"

# ---------------------------- Main menu actions ----------------------------

def act_evaluate():
    clr()
    print(box("Evaluate f(...)", [
        "Enter an expression, then provide variable values.",
        "Supported: + - * / ^, parentheses, sin cos tan exp ln sqrt abs.",
        "Constants: pi, e",
        "Example: sin(x)^2 + cos(x)^2"
    ]))
    expr = input("\nExpression f = ").strip()
    f = parse(expr)
    vars_raw = input("Variables used (comma-separated, e.g. x,y) = ").strip()
    vars_list = parse_vars_list(vars_raw)
    env = prompt_vars(vars_list)
    val = eval_ast(f, env)
    print("\n" + box("Result", ["f = " + to_str(f), "value = " + str(val)]))
    input("\nPress Enter to return...")

def act_symbolic_derivative():
    clr()
    print(box("Symbolic derivative d/dx", [
        "Computes a symbolic derivative and applies light simplification.",
        "Tip: tan(u) is rewritten as sin(u)/cos(u) automatically.",
        "Example: d/dx of x^3 + 2*x -> 3*x^2 + 2"
    ]))
    expr = input("\nExpression f = ").strip()
    var = input("Differentiate w.r.t variable = ").strip()
    f = parse(expr)
    df = simplify(deriv(f, var))
    print("\n" + box("Derivative", ["df/d" + var + " = " + to_str(df)]))
    input("\nPress Enter to return...")

def act_partial_derivative():
    clr()
    print(box("Partial derivative ∂f/∂x", [
        "Enter a multivariable expression and a variable name.",
        "Example: f = x^2*y + exp(z) -> ∂f/∂y = x^2"
    ]))
    expr = input("\nExpression f = ").strip()
    var = input("Partial w.r.t variable = ").strip()
    f = parse(expr)
    df = simplify(deriv(f, var))
    print("\n" + box("Partial derivative", ["∂f/∂" + var + " = " + to_str(df)]))
    input("\nPress Enter to return...")

def act_gradient():
    clr()
    print(box("Gradient ∇f", [
        "Provide scalar field f and variables list (e.g., x,y,z).",
        "Output is [∂f/∂x, ∂f/∂y, ...]."
    ]))
    expr = input("\nScalar field f = ").strip()
    vars_list = parse_vars_list(input("Variables (comma-separated) = ").strip())
    f = parse(expr)
    g = gradient(f, vars_list)
    print("\n" + box("Gradient", ["∇f = " + pretty_vector(g)]))
    input("\nPress Enter to return...")

def act_jacobian():
    clr()
    print(box("Jacobian J", [
        "Provide vector function F and variables list.",
        "You'll enter components one by one.",
        "Output is matrix Jij = ∂Fi/∂xj."
    ]))
    m = int(input("\nNumber of components in F = ").strip())
    comps = []
    i = 0
    while i < m:
        comps.append(parse(input("F" + str(i+1) + " = ").strip()))
        i += 1
    vars_list = parse_vars_list(input("Variables (comma-separated) = ").strip())
    J = jacobian(comps, vars_list)
    print("\n" + box("Jacobian", ["J = " + pretty_matrix(J)]))
    input("\nPress Enter to return...")

def act_hessian():
    clr()
    print(box("Hessian H", [
        "Provide scalar field f and variables list.",
        "Output is matrix Hij = ∂²f/∂xi∂xj."
    ]))
    expr = input("\nScalar field f = ").strip()
    vars_list = parse_vars_list(input("Variables (comma-separated) = ").strip())
    f = parse(expr)
    H = hessian(f, vars_list)
    print("\n" + box("Hessian", ["H = " + pretty_matrix(H)]))
    input("\nPress Enter to return...")

def act_divergence_curl():
    clr()
    print(box("Divergence ∇·F and Curl ∇×F (3D)", [
        "Divergence: ∂Fx/∂x + ∂Fy/∂y (+ ∂Fz/∂z in 3D).",
        "Curl (3D): [∂Fz/∂y-∂Fy/∂z, ∂Fx/∂z-∂Fz/∂x, ∂Fy/∂x-∂Fx/∂y].",
        "Enter variables as x,y,z for curl."
    ]))
    dim = int(input("\nDimension (2 or 3) = ").strip())
    if dim not in (2,3):
        raise ValueError("Dimension must be 2 or 3")
    vars_list = parse_vars_list(input("Variables (comma-separated, length " + str(dim) + ") = ").strip())
    if len(vars_list) != dim:
        raise ValueError("You must provide exactly " + str(dim) + " variables")
    comps = []
    i = 0
    while i < dim:
        comps.append(parse(input("F_" + vars_list[i] + " = ").strip()))
        i += 1
    divv = divergence(comps, vars_list)
    lines = ["∇·F = " + to_str(divv)]
    if dim == 3:
        c = curl_3d(comps, vars_list)
        lines.append("∇×F = " + pretty_vector(c))
    print("\n" + box("Result", lines))
    input("\nPress Enter to return...")

def act_num_derivative():
    clr()
    print(box("Numerical derivative", [
        "Central difference: f'(x) ≈ (f(x+h)-f(x-h)) / (2h).",
        "Good default: h = 1e-5"
    ]))
    expr = input("\nExpression f = ").strip()
    var = input("Differentiate w.r.t variable = ").strip()
    x0 = float(input("Point " + var + " = ").strip())
    h = float(input("Step h (e.g. 1e-5) = ").strip())
    other_vars = input("Other variables (comma-separated, or blank) = ").strip()
    env = {}
    if other_vars.strip() != "":
        vars_list = parse_vars_list(other_vars)
        env = prompt_vars(vars_list)
    f = parse(expr)
    val = num_derivative(f, var, x0, h, env)
    print("\n" + box("Result", ["f'(" + str(x0) + ") ≈ " + str(val)]))
    input("\nPress Enter to return...")

def act_integral_1d():
    clr()
    print(box("Definite integral ∫ f(x) dx (numerical)", [
        "Methods: trapezoid, simpson, adaptive simpson.",
        "Simpson is usually good for smooth functions.",
        "Adaptive Simpson uses an error tolerance eps."
    ]))
    expr = input("\nIntegrand f = ").strip()
    var = input("Integration variable = ").strip()
    a = float(input("Lower bound a = ").strip())
    b = float(input("Upper bound b = ").strip())
    other_vars = input("Other variables (comma-separated, or blank) = ").strip()
    env = {}
    if other_vars.strip() != "":
        vars_list = parse_vars_list(other_vars)
        env = prompt_vars(vars_list)

    method = input("Method (trap/simpson/adapt) = ").strip().lower()
    f = parse(expr)
    if method in ('trap', 'trapezoid'):
        n = int(input("Subintervals n (e.g. 1000) = ").strip())
        val = trapz_1d(f, var, a, b, n, env)
    elif method in ('simpson', 'simp'):
        n = int(input("Subintervals n (even preferred, e.g. 200) = ").strip())
        val = simpson_1d(f, var, a, b, n, env)
    elif method in ('adapt', 'adaptive'):
        eps = float(input("Tolerance eps (e.g. 1e-8) = ").strip())
        maxd = int(input("Max recursion depth (e.g. 20) = ").strip())
        val = adaptive_simpson_1d(f, var, a, b, eps, env, maxd)
    else:
        raise ValueError("Unknown method")
    print("\n" + box("Integral", ["∫ f d" + var + " from " + str(a) + " to " + str(b) + " ≈ " + str(val)]))
    input("\nPress Enter to return...")

def act_symbolic_integral():
    clr()
    print(box("Symbolic indefinite integral ∫ f dx (limited)", [
        "Rule-based symbolic ∫ for common forms:",
        "  polynomials, 1/x, 1/(a*x+b), sin/cos/exp with linear inner.",
        "If it can't match a rule, it will show an error.",
        "Tip: use numerical ∫ when symbolic ∫ fails."
    ]))
    expr = input("\nIntegrand f = ").strip()
    var = input("Integrate w.r.t variable = ").strip()
    f = parse(expr)
    F = simplify(integrate(f, var))
    print("\n" + box("Antiderivative", ["∫ f d" + var + " = " + to_str(F) + " + C"]))
    input("\nPress Enter to return...")

def act_integral_2d():
    clr()
    print(box("Double integral ∬ f(x,y) dy dx (nested Simpson)", [
        "Computes ∬ f(x,y) dy dx over rectangle [a,b]×[c,d].",
        "Requires two variables; subdivisions will be made even if needed."
    ]))
    expr = input("\nIntegrand f = ").strip()
    varx = input("x-variable name (e.g. x) = ").strip()
    vary = input("y-variable name (e.g. y) = ").strip()
    a = float(input("x lower a = ").strip())
    b = float(input("x upper b = ").strip())
    c = float(input("y lower c = ").strip())
    d = float(input("y upper d = ").strip())
    nx = int(input("x subdivisions nx (e.g. 60) = ").strip())
    ny = int(input("y subdivisions ny (e.g. 60) = ").strip())
    other_vars = input("Other variables (comma-separated, or blank) = ").strip()
    env = {}
    if other_vars.strip() != "":
        vars_list = parse_vars_list(other_vars)
        env = prompt_vars(vars_list)
    f = parse(expr)
    val = simpson_2d(f, varx, a, b, nx, vary, c, d, ny, env)
    print("\n" + box("Double integral", [
        "∬ f d" + vary + " d" + varx + " over ["+str(a)+","+str(b)+"]×["+str(c)+","+str(d)+"] ≈",
        str(val)
    ]))
    input("\nPress Enter to return...")

def act_help():
    clr()
    lines = [
        "Expression syntax:",
        "  - Operators: +  -  *  /  ^  (power)",
        "  - Parentheses: ( )",
        "  - Functions: sin(x), cos(x), tan(x), exp(x), ln(x), sqrt(x), abs(x)",
        "  - Constants: pi, e",
        "",
        "Tips:",
        "  - Use explicit multiplication: 2*x (not 2x).",
        "  - For tan(u), derivative is handled by rewriting tan(u)=sin(u)/cos(u).",
        "  - Symbolic ∫ uses limited rules; use numerical ∫ when it fails.",
        "  - ln(x) needs x>0; sqrt(x) needs x>=0.",
        "",
        "Example inputs:",
        "  f = x^3 + 2*x",
        "  f = sin(x)^2 + cos(x)^2",
        "  f = x^2*y + exp(z)",
        "  Vector field: Fx=x*y, Fy=y*z, Fz=z*x"
    ]
    print(box("Help", lines))
    input("\nPress Enter to return...")

def main():
    while True:
        clr()
        header = color("MiniCalc (no imports) — Calculus & Multivariable Toolbox", "1;36")
        print(header)
        print("Date/time shown by your terminal.\n")
        menu = [
            "1) Evaluate expression",
            "2) Symbolic derivative d/dx",
            "3) Partial derivative ∂f/∂x",
            "4) Gradient ∇f",
            "5) Jacobian J",
            "6) Hessian H",
            "7) Divergence ∇·F / Curl ∇×F (3D)",
            "8) Numerical derivative",
            "9) Definite integral (1D)",
            "10) Symbolic integral (indefinite)",
            "11) Double integral (2D rectangle)",
            "H) Help",
            "Q) Quit"
        ]
        print(box("Main Menu", menu))
        choice = input("\nSelect an option: ").strip().lower()
        try:
            if choice == '1': act_evaluate()
            elif choice == '2': act_symbolic_derivative()
            elif choice == '3': act_partial_derivative()
            elif choice == '4': act_gradient()
            elif choice == '5': act_jacobian()
            elif choice == '6': act_hessian()
            elif choice == '7': act_divergence_curl()
            elif choice == '8': act_num_derivative()
            elif choice == '9': act_integral_1d()
            elif choice == '10': act_symbolic_integral()
            elif choice == '11': act_integral_2d()
            elif choice == 'h': act_help()
            elif choice == 'q':
                clr()
                print("Bye!")
                return
            else:
                print("\nUnknown option.")
                input("Press Enter...")
        except Exception as e:
            print("\n" + box("Error", [str(e)]))
            input("\nPress Enter to return to menu...")

if __name__ == "__main__":
    main()
