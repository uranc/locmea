class newLFPySimulation(object):
"""Class of LFPy experiment instance:
Attributes:
cellParameters
electrodeParameters
"""


def __init__(self, cellParameters={
        'morphology': 'L5bPCmodelsEH/morphologies/cell1.asc',
        'templatefile': ['L5bPCmodelsEH/models/L5PCbiophys3.hoc',
                         'L5bPCmodelsEH/models/L5PCtemplate.hoc'],
        'templatename': 'L5PCtemplate',
        'templateargs': 'L5bPCmodelsEH/morphologies/cell1.asc',
        'passive': False,
        'nsegs_method': None,
        'timeres_NEURON': 2 ** -6,
        'timeres_python': 2 ** -6,
        'tstartms': -159,
        'tstopms': 10,
        'v_init': -60,
        'pt3d': True},
        electrodeParameters={'see': 4}):
"""Returns a newLFPySimulation object with cell parameters
*cellParameters"""
self.cellParam = cellParameters
self.electrodeParameters = electrodeParameters


def simulate_data(self):
"""Prints bla bla. Returns blabla"""
print "simulate data now"
