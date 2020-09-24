
units_table = {'Hz': 1, 'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9, 'THz': 1e12, 
               's': 1, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9,
               'shot': 1}

def process_units(x_data, units):
    try:
        factor = units_table[units]
        if type(x_data).__name__ == 'list':
            return [i/factor for i in x_data]
        else:
            return x_data/factor
        
    except Exception as e:
        print('Process units error: {}'.format(e))
        return x_data