"""
Copyright (C) This program is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version. This program is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
License for more details.
--------------------------------------------------------------------
Author: Cem Uran <cem.uran@uranus.uni-freiburg.de>
--------------------------------------------------------------------
Inverse optimizer example
--------------------------------------------------------------------
"""
from setOptProb import opt_out


class param_opt(opt_out):
    """
    Nice class info
    """
    def __init__(self, *args, **kwargs):
        """
        use data for parameter optimization
        """
        print "hello"
