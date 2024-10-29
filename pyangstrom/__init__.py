"""A library for performing the Angstrom method on thermal video data.

Please consult the PyAngstrom repo for installation instructions and the wiki
for detailed usage instructions. Functions for basic and NTRG usage have been
preimported.

Functions
---------
- analyze_recording: https://ruralbrick.github.io/Angstrom-Method-Rewrite/pyangstrom/pipeline.html#pyangstrom.pipeline.analyze_recording
- hu_batch_process: https://ruralbrick.github.io/Angstrom-Method-Rewrite/pyangstrom/wrappers/pipeline.html#pyangstrom.wrappers.pipeline.hu_batch_process

References
----------
- PyAngstrom Repository: https://github.com/RuralBrick/Angstrom-Method-Rewrite
- PyAngstrom Wiki: https://github.com/RuralBrick/Angstrom-Method-Rewrite/wiki
"""
from pyangstrom.pipeline import analyze_recording
from pyangstrom.wrappers.pipeline import hu_batch_process
