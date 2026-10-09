# Attribution

`sehope/vendor/{hierarchical,block,layers,common}.py` are adapted from
https://github.com/PSRben/VisionHOPE at
`98cfb5d6004472a0ba3d212d3cf884393e39c134`, under the MIT License.
The modifications remove timm and optional CUDA utility imports while preserving
native parameter names, shapes, and unfused math. `operator.py` and `recurrence.py`
are portable reimplementations of the same repository's operator and chunk kernels.
The original license is included as `LICENSE.VisionHOPE`.

No pretrained weight or dataset redistribution is included. Download weights using
the pinned metadata in `weights_manifest.json` and follow each dataset's terms.
