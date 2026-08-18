"""Deployment code extracted from research repositories.

Each subpackage is one model's inference path, reduced from its upstream project to the forward
pass, the modules it touches, the weight loader and the pre/post-processing maths -- and left
otherwise unedited, so it can still be diffed against the commit it came from. Its ``PROVENANCE``
records which commit that is.

Nothing here reaches outside itself: no vendor downloads weights, chooses a device, or knows what
mozo is. Fetching is :mod:`mozo.weights`, naming is :mod:`mozo.labels`, and execution is
:mod:`mozo.runtimes`. A vendor is handed a path and asked for numbers.
"""
