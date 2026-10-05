#!/usr/bin/env python3
"""Public helper facade for the subliminal-learning Station task.

Keep importing from this module in submissions. Implementation lives in the
`subliminal` package so the public API remains stable while internals stay
maintainable.

Module map:
- `subliminal.core`: constants, storage paths, JSON/JSONL I/O, prompt helpers,
  canonical eval config, and small statistical utilities.
- `subliminal.reference`: bundled reference datasets/results, reference metric
  availability, and cat-vs-regular comparison helpers.
- `subliminal.validation`: numeric anti-leakage checks and row filtering.
- `subliminal.summary`: canonical P(target) evaluation summarization.
- `subliminal.manifests`: trained-adapter manifest read/write helpers.
- `subliminal.diagnostics`: digit, format, number n-gram, transition, and
  high-order dataset diagnostics.
- `subliminal.interventions`: number shuffling, canonicalization, and
  bucketization interventions.
- `subliminal.generation`: numeric dataset generation configs and adequacy
  reports.
- `subliminal.training`: Unsloth LoRA training helpers and adapter-manifest
  creation.
- `subliminal.evaluation`: canonical evaluation for trained runs and existing
  adapters.
- `subliminal.workflows`: convenience end-to-end entry points and summary
  emission.
"""

from __future__ import annotations

from subliminal.core import *
from subliminal.validation import *
from subliminal.summary import *
from subliminal.manifests import *
from subliminal.reference import *
from subliminal.diagnostics import *
from subliminal.interventions import *
from subliminal.generation import *
from subliminal.training import *
from subliminal.evaluation import *
from subliminal.workflows import *
