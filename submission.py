from collections import defaultdict
import itertools
import re
import numpy as np

# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
	k = 1
	symbols_info = read_symbol(Symbol_File)
	states_info = read_state(State_File)
	s_to_id, id_to_s = translator(symbols_info['symbols'])
	querys = read_query(Query_File)
	P = p_state_to_state(states_info['number_of_state'], states_info['state_n'])
	Q = p_state_to_symbol(states_info['number_of_state'], symbols_info['number_of_symbol'], symbols_info['symbol_n'])
	querys = [[s_to_id[sym] if sym in s_to_id else symbols_info['number_of_symbol'] for sym in query] for query in querys]
	output = []
	for query in querys:
		records, state_P = pred(P,Q, query, k, states_info['number_of_state'])
		state_P = np.log(state_P)
		output.append(records[0] + [state_P[0]])
	return output



# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
	symbols_info = read_symbol(Symbol_File)
	states_info = read_state(State_File)
	s_to_id, id_to_s = translator(symbols_info['symbols'])
	querys = read_query(Query_File)
	P = p_state_to_state(states_info['number_of_state'], states_info['state_n'])
	Q = p_state_to_symbol(states_info['number_of_state'], symbols_info['number_of_symbol'], symbols_info['symbol_n'])
	querys = [[s_to_id[sym] if sym in s_to_id else symbols_info['number_of_symbol'] for sym in query] for query in querys]
	output = []
	for query in querys:
		records, state_P = pred(P,Q, query, k, states_info['number_of_state'])
		state_P = np.log(state_P)
		for i in range(len(state_P)):
			output.append(records[i] + [state_P[i]])
	return output


# Question 3 + Bonus
def read_symbol_adv(Symbol_File):
    output = []
    with open(Symbol_File, 'r') as f:
        f = list(f)
        old_number_of_symbol = int(f[0])
        old_symbols = [x.strip() for x in f[1:1 + old_number_of_symbol]]
        symbols_functions = [
            ('lot', lambda x: 'lot' == x.lower()[0:3]),
            ('lvl/ste/shp', lambda x: number_of_digit(x) != 0 and (
                x.lower()[0:3] in ['ste', 'shp', 'lvl', 'apr'] or
                'flr' in x.lower() or
                (x[0] == 'L' and number_of_digit(x) == len(x) - 1)
            )),
            ('kiosk', lambda x: x.lower() == 'kiosk'),
            ('14inside', lambda x : x.lower() in ['community', 'centre', 'suites', 'suite', 'institute', 'estate', 'postal', 'branch']),
            ('unit', lambda x: number_of_digit(x) != 0 and (x[0] == 'U')),
            ('st/rd', lambda x: x.lower() in ['st', 'rd', 'ave', 'drv', 'hwy', 'la']),
        ]
        symbols, functions = zip(*symbols_functions)
        symbols = list(symbols)
        
        old_id_to_new_id = {}
        for i, x in enumerate(old_symbols):
            changed = False
            for fs_i,fs_f in enumerate(functions):
                if not changed and fs_f(x):
                    old_id_to_new_id[i] = fs_i
                    changed = True
            if not changed:
                old_id_to_new_id[i] = len(symbols)
                symbols.append(x)

        number_of_symbol = len(symbols)
        symbol_n = defaultdict(int)
        for row in f[1+old_number_of_symbol:]:
            row = [int(x.strip()) for x in row.split()]
            symbol_n[row[0], old_id_to_new_id[row[1]]] += row[2]
        
    s_to_old_id, old_id_to_s = translator(old_symbols)
    
    def sym_to_id(sym):
        for fs_i,fs_f in enumerate(functions):
            if fs_f(sym):
                return fs_i
        return old_id_to_new_id[s_to_old_id[sym]] if sym in s_to_old_id else number_of_symbol
    
    return {
        'old_symbols':old_symbols,
        'symbols':symbols,
        'symbol_n':symbol_n,
        'number_of_symbol':number_of_symbol,
        'old_number_of_symbol':old_number_of_symbol,
        'old_id_to_new_id':old_id_to_new_id,
        'sym_to_id': sym_to_id,
        's_to_old_id':s_to_old_id,
        'old_id_to_s':old_id_to_s
    }

def number_of_digit(string):
    return len(list(filter(str.isdigit, string)))

def pred_adv(P, Q, query, k, number_of_state, id_to_s): #works
    Fs = [get_F2(o, P ,Q) for o in query]
    indice_with_number = [0,1,6,11]
    for i,o in zip(range(len(query) - 3), query[:-3]):
        if o in id_to_s and number_of_digit(id_to_s[o]) == 0:
            continue
            Fs[i][:,indice_with_number] = 0
            
    records = [[[number_of_state - 2]] for j in range(number_of_state)]
    states_P = initial_state(number_of_state)
    for F in Fs:
        new_state = []
        new_record = []
        for state_F, i in zip(F.T, range(number_of_state)): #state_F.shape == (5,)
            if i in [number_of_state - 1, number_of_state - 2 ]:
                w = len(new_state[0])
                new_state.append(np.zeros(w))
                new_record.append([[] for j in range(w)])
            else:
                _prob_o = np.multiply(states_P.T, state_F).T #_prob_o.shape == (5,1)
                top_k_index =  argsort2d(_prob_o, k)  #len(top_k_index) == k, top_k_index[0] = (x,y)
                new_state_row = _prob_o.take(top_k_index)
                new_record_row = [ records[j//len(records[0])][j%len(records[0])] + [i] for j in top_k_index]
                new_state.append(new_state_row)
                new_record.append(new_record_row)
        states_P = np.array(new_state)
        records = new_record
    states_P = np.multiply(states_P.T, P[:, number_of_state - 1]).T
    top_k_index =  argsort2d(states_P, k)
    output_states = states_P.take(top_k_index)
    output_records = [ records[j//len(records[0])][j%len(records[0])] + [number_of_state - 1] for j in top_k_index]
    return output_records, output_states

def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
	k = 1
	state_alpha = 0.5
	symbol_alpha = 1.0
	symbols_info = read_symbol_adv(Symbol_File)
	states_info = read_state(State_File)
	s_to_id, id_to_s = translator(symbols_info['old_symbols'])
	querys = read_query(Query_File)

	#hard_code observation
	symbols_info['symbol_n'][(1,0)] = 2000
	symbols_info['symbol_n'][(9,1)] = 2000
	symbols_info['symbol_n'][(12,2)] = 1000
	symbols_info['symbol_n'][(14,3)] = 2000

	querys_tokens = [
	    [symbols_info['sym_to_id'](sym) for sym in query]
	    for query in querys
	]

	output = []
	P = p_state_to_state(states_info['number_of_state'], states_info['state_n'], state_alpha)
	Q = p_state_to_symbol(states_info['number_of_state'], symbols_info['number_of_symbol'], symbols_info['symbol_n'], symbol_alpha)

	for query in querys_tokens:
	    records, state_P = pred_adv(P,Q, query, k, states_info['number_of_state'], id_to_s)
	    state_P = np.log(state_P)
	    r0 = records[0]
	    s0 = state_P[0]
	    output.append(r0 + [s0])
	return output

def translator(symbols):
	length = len(symbols)
	symbol_to_id = dict(zip(symbols, range(length)))
	id_to_symbol = dict(zip(range(length), symbols))
	return symbol_to_id, id_to_symbol
    
def read_symbol(Symbol_File):
	output = []
	with open(Symbol_File, 'r') as f:
	    f = list(f)
	    number_of_symbol = int(f[0])
	    symbols = [x.strip() for x in f[1:1 + number_of_symbol]]
	    symbol_n = defaultdict(int)
	    for row in f[1+number_of_symbol:]:
	        row = [int(x.strip()) for x in row.split()]
	        symbol_n[row[0], row[1]] += row[2]
	return {'symbols':symbols, 'symbol_n':symbol_n, 'number_of_symbol':number_of_symbol}
	# {symbols: [], symbol_p:symbol_p}
    
def read_query(Query_File):
	output = []
	with open(Query_File, 'r') as f:
	    for row in f:
	        indices = [x.span() for x in re.finditer(r"[\,\(\)\/\-\&]", row)]
	        indices = [0] + list(itertools.chain.from_iterable(indices)) + [len(row)]
	        _output = []
	        for i in range(len(indices) - 1):
	            _output += [x.strip() for x in row[indices[i]: indices[i+1]].split(' ') if x.strip() != '']
	        output.append(_output)
	return output
	#return [
	# [...],...
	#]

def read_state(State_File):
	output = []
	with open(State_File, 'r') as f:
	    f = list(f)
	    number_of_state = int(f[0])
	    states = [x.strip() for x in f[1:1 + number_of_state]]
	    state_n = defaultdict(int)
	    for row in f[1+number_of_state:]:
	        row = [int(x.strip()) for x in row.split()]
	        state_n[row[0], row[1]] += row[2]
	return {'states':states, 'state_n':state_n, 'number_of_state':number_of_state}
	# {symbols: [], state_p:state_p}

def p_state_to_state(number_of_state, state_n, alpha = 1.0):
	matrix = np.zeros((number_of_state, number_of_state))
	for src_state, des_state in state_n:
	    matrix[src_state, des_state] = state_n[(src_state, des_state)]
	prior = np.sum(matrix, axis=1) + (number_of_state - 1)*alpha
	base = np.divide(matrix.T + alpha, prior).T
	base[:, number_of_state - 2] = 0
	base[number_of_state - 1, :] = 0
	return base

def p_state_to_symbol(number_of_state, number_of_symbol, symbol_n, alpha = 1.0):
	matrix = np.zeros((number_of_state, number_of_symbol + 1)) # +1 for 'unknown' symbol
	for state, symbol in symbol_n:
	    matrix[state, symbol] = symbol_n[(state, symbol)]
	prior = np.sum(matrix, axis=1) + (number_of_symbol + 1)*alpha
	return np.divide(matrix.T + alpha, prior).T

def initial_state(number_of_state):
	state = np.zeros(number_of_state).reshape((-1,1))
	state[number_of_state - 2][0] = 1
	return state

def get_F(o, P, Q):
	_Q = Q[:, o]
	n_state = len(P)
	output = np.zeros((n_state, n_state)).T
	for i in range(n_state):
	    for j in range(n_state):
	        output[(i,j)] = P[i,j]*_Q[j]
	return output

def get_F2(o, P, Q):
	return np.multiply(P, Q[:, o])

def argsort2d(matrix, k = 3):
	h, _ = matrix.shape
	sorting = (-matrix.reshape(-1)).argsort(kind='mergesort')[:k]
	return [x for x in sorting if matrix.reshape(-1)[x] != 0]

def pred(P, Q, query, k, number_of_state):
	Fs = [get_F2(o,P,Q) for o in query]
	records = [[[number_of_state - 2]] for j in range(number_of_state)]
	states_P = initial_state(number_of_state)
	for F in Fs:
	    new_state = []
	    new_record = []
	    for state_F, i in zip(F.T, range(number_of_state)): #state_F.shape == (5,)
	        if i in [number_of_state - 1, number_of_state - 2 ]:
	            w = len(new_state[0])
	            new_state.append(np.zeros(w))
	            new_record.append([[] for j in range(w)])
	        else:
	            _prob_o = np.multiply(states_P.T, state_F).T #_prob_o.shape == (5,1)
	            top_k_index =  argsort2d(_prob_o, k)  #len(top_k_index) == k, top_k_index[0] = (x,y)
	            new_state_row = _prob_o.take(top_k_index)
	            new_record_row = [ records[j//len(records[0])][j%len(records[0])] + [i] for j in top_k_index]
	            new_state.append(new_state_row)
	            new_record.append(new_record_row)
	    states_P = np.array(new_state)
	    records = new_record
	states_P = np.multiply(states_P.T, P[:, number_of_state - 1]).T
	top_k_index =  argsort2d(states_P, k)
	output_states = states_P.take(top_k_index)
	output_records = [ records[j//len(records[0])][j%len(records[0])] + [number_of_state - 1] for j in top_k_index]
	return output_records, output_states


if __name__ == '__main__':
	k = 2
	output = top_k_viterbi('./State_File', './Symbol_File', './Query_File', k)
	for i  in range(0, len(output)// k):
		print(f'Answer for the query {i}')
		print('\n'.join([str(x) for x in output[i*k : (i+1)*k]]))

