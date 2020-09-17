"""
Module with invoke tasks
"""

import invoke

import net.invoke.analysis
import net.invoke.docker
import net.invoke.visualize
import net.invoke.tests

# Default invoke collection
ns = invoke.Collection()

# Add collections defined in other files
ns.add_collection(net.invoke.analysis)
ns.add_collection(net.invoke.docker)
ns.add_collection(net.invoke.visualize)
ns.add_collection(net.invoke.tests)
