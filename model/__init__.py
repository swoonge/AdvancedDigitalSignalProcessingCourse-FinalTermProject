from .ConvCell import *
from .Sign import *
from .network import *
from .encoder import *
from .decoder import *
# my_variable = 10

__all__ = ['ConvLSTMCell', 'ConvGRUCell', 'Sign', 'network', 'encoder', 'decoder']

def __version__():
    return '0.1.0'

def __info__():
    return 'This is the model package of the AdvancedDigitalSignalProcessingCourse-FinalTermProject project.'