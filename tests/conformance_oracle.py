"""Independent exact-arithmetic checks; deliberately imports no ModeBench code."""
import ast
from collections import Counter
from fractions import Fraction
from itertools import combinations, permutations, product


def arithmetic(text):
    numbers = []
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            numbers.append(node.value)
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        assert isinstance(node, ast.BinOp)
        a, b = visit(node.left), visit(node.right)
        if isinstance(node.op, ast.Add): return a + b
        if isinstance(node.op, ast.Sub): return a - b
        if isinstance(node.op, ast.Mult): return a * b
        if isinstance(node.op, ast.Div): return a / b
        raise AssertionError('unsupported arithmetic')
    value = visit(ast.parse(text, mode='eval').body)
    return value, Counter(numbers)


def countdown_solutions(spec):
    def trees(numbers):
        if len(numbers) == 1:
            yield Fraction(numbers[0]), str(numbers[0])
        for cut in range(1, len(numbers)):
            for a, x in trees(numbers[:cut]):
                for b, y in trees(numbers[cut:]):
                    for op, value in [('+', a+b), ('-', a-b), ('*', a*b)]:
                        yield value, f'({x}{op}{y})'
                    if b:
                        yield a/b, f'({x}/{y})'
    for numbers in sorted(set(permutations(spec['numbers']))):
        for value, expression in trees(numbers):
            if value == spec['target']:
                yield expression


def graph_valid(spec, colors):
    return (len(colors) == spec['n'] and all(c in (1,2,3) for c in colors)
            and all(colors[a-1] != colors[b-1] for a,b in spec['edges'])
            and all(c is None or colors[i] == c for i,c in enumerate(spec.get('partial_colors', [None]*spec['n']))))


def graph_solutions(spec):
    for colors in product((1,2,3), repeat=spec['n']):
        if graph_valid(spec, colors):
            yield ''.join(map(str, colors))


def factor_program(spec, which):
    outputs = [[d for d in range(2,n) if n%d == 0][which] for n in spec['cases']]
    text = str(outputs[-1])
    for n,d in reversed(list(zip(spec['cases'][:-1], outputs[:-1]))):
        text = f'{d} if n=={n} else {text}'
    return 'lambda n: '+text, outputs


def pantry_valid(spec, allocations):
    byid = {i['id']: i for i in spec['ingredients']}
    if not spec['min_ingredients'] <= len(allocations) <= spec['max_ingredients']:
        return False
    totals = Counter()
    for name, grams in allocations.items():
        if name not in byid: return False
        i = byid[name]
        if set(i['tags']) & set(spec['forbidden_tags']): return False
        if not i['min_if_used_g'] <= grams <= i['available_g'] or grams % i['step_g']: return False
        totals['mass_g'] += grams
        for attr, value in i['attributes_per_100g'].items():
            totals[attr] += Fraction(str(value))*grams/100
    return all(('min' not in bounds or totals[attr] >= Fraction(str(bounds['min'])))
               and ('max' not in bounds or totals[attr] <= Fraction(str(bounds['max'])))
               for attr,bounds in spec['targets'].items())


def pantry_solutions(spec):
    for count in range(spec['min_ingredients'], spec['max_ingredients']+1):
        for group in combinations(spec['ingredients'],count):
            grids = [range(i['min_if_used_g'],i['available_g']+1,i['step_g']) for i in group]
            for quantities in product(*grids):
                allocation = dict(zip([i['id'] for i in group],quantities))
                if pantry_valid(spec,allocation):
                    yield allocation
                    break


def linear_expression(text, bindings):
    """Evaluate the restricted reference grammar as ax+b over exact rationals."""
    def visit(node):
        if isinstance(node,ast.Name):
            return (Fraction(1),Fraction(0)) if node.id=='x' else (Fraction(0),Fraction(str(bindings[node.id])))
        if isinstance(node,ast.Constant): return Fraction(0),Fraction(node.value)
        assert isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
        values=[visit(n) for n in node.args]
        op=node.func.id
        a,b=values[0]
        if op=='neg': return -a,-b
        c,d=values[1]
        if op=='add': return a+c,b+d
        if op=='sub': return a-c,b-d
        if op=='mul':
            assert not (a and c), 'nonlinear equation'
            return a*d+b*c,b*d
        if op=='div':
            assert not c and d, 'nonconstant or zero divisor'
            return a/d,b/d
        raise AssertionError(op)
    return visit(ast.parse(text,mode='eval').body)


def mathir_trace(spec, program):
    lhs=linear_expression(spec['initial_lhs'],spec['bindings'])
    rhs=linear_expression(spec['initial_rhs'],spec['bindings'])
    target=(rhs[1]-lhs[1])/(lhs[0]-rhs[0])
    trace=[]
    for action in program.split(';'):
        command=spec['actions'][action]
        op,arg=command.split('(',1)
        a,b=linear_expression(arg[:-1],spec['bindings'])
        if op in ('mul','div') and a:
            raise AssertionError('nonconstant scaling')
        def apply(side):
            x,y=side
            if op=='add': return x+a,y+b
            if op=='sub': return x-a,y-b
            if op=='mul': return x*b,y*b
            if op=='div': return x/b,y/b
            raise AssertionError(op)
        lhs,rhs=apply(lhs),apply(rhs)
        assert lhs[0]-rhs[0], 'degenerate equation'
        assert (rhs[1]-lhs[1])/(lhs[0]-rhs[0])==target
        trace.append((lhs,rhs))
    assert (lhs==(1,0) and rhs==(0,target)) or (rhs==(1,0) and lhs==(0,target)), 'x not isolated'
    return trace


def mathir_solutions(spec):
    for length in range(1,spec['max_steps']+1):
        for steps in product(spec['actions'],repeat=length):
            program=';'.join(steps)
            try: mathir_trace(spec,program)
            except (AssertionError,ZeroDivisionError): continue
            yield program


def verify_witness(spec, domain, witness):
    """Review fixture validity without calling production graders or key builders."""
    if domain=='countdown':
        value,numbers=arithmetic(witness['response'])
        assert value==spec['target'] and numbers==Counter(spec['numbers'])
    elif domain=='graph_coloring':
        assert graph_valid(spec,list(map(int,witness['response'])))
    elif domain=='python_factors':
        assert all(1<d<n and n%d==0 for n,d in zip(spec['cases'],witness['outputs']))
        # Evaluate only the fixture's literal conditional grammar, without eval.
        tree=ast.parse(witness['response'],mode='eval').body
        def visit(node,n):
            if isinstance(node,ast.Constant): return node.value
            assert isinstance(node,ast.IfExp)
            test=node.test
            assert isinstance(test,ast.Compare) and isinstance(test.left,ast.Name) and test.left.id=='n'
            assert len(test.ops)==1 and isinstance(test.ops[0],ast.Eq)
            return visit(node.body if n==test.comparators[0].value else node.orelse,n)
        assert [visit(tree.body,n) for n in spec['cases']]==witness['outputs']
    elif domain=='pantry_plan':
        assert pantry_valid(spec,witness['allocations'])
    else:
        mathir_trace(spec,witness['response'])


def independent_mode(spec, domain, witness):
    """Check mode distinctions without importing canonicalization code."""
    if domain=='graph_coloring': return tuple(map(int,witness['response']))
    if domain=='python_factors': return tuple(witness['outputs'])
    if domain=='pantry_plan': return tuple(sorted(witness['allocations']))
    if domain=='countdown':
        def tree(node):
            if isinstance(node,ast.Constant): return ('number',node.value)
            if isinstance(node,ast.UnaryOp): return (type(node.op).__name__,tree(node.operand))
            a,b=tree(node.left),tree(node.right)
            if isinstance(node.op,(ast.Add,ast.Mult)):
                a,b=sorted((a,b),key=repr)
            return (type(node.op).__name__,a,b)
        return tree(ast.parse(witness['response'],mode='eval').body)
    import sympy as sp
    symbols={name:sp.Symbol(name) for name in (*spec['bindings'],'x')}
    def expression(text):
        def visit(node):
            if isinstance(node,ast.Name): return symbols[node.id]
            if isinstance(node,ast.Constant): return sp.Rational(node.value)
            args=[visit(a) for a in node.args]
            op=node.func.id
            if op=='add': return args[0]+args[1]
            if op=='sub': return args[0]-args[1]
            if op=='mul': return args[0]*args[1]
            if op=='div': return args[0]/args[1]
            if op=='neg': return -args[0]
            raise AssertionError(op)
        return visit(ast.parse(text,mode='eval').body)
    lhs,rhs=expression(spec['initial_lhs']),expression(spec['initial_rhs'])
    trace=[]
    for action in witness['response'].split(';'):
        op,arg=spec['actions'][action].split('(',1)
        value=expression(arg[:-1])
        if op=='add': lhs,rhs=lhs+value,rhs+value
        if op=='sub': lhs,rhs=lhs-value,rhs-value
        if op=='mul': lhs,rhs=lhs*value,rhs*value
        if op=='div': lhs,rhs=lhs/value,rhs/value
        lhs,rhs=sp.cancel(lhs),sp.cancel(rhs)
        trace.append((lhs,rhs))
    assert (lhs==symbols['x'] and symbols['x'] not in rhs.free_symbols) or (rhs==symbols['x'] and symbols['x'] not in lhs.free_symbols)
    return tuple(trace)
