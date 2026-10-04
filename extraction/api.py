"""Public entry point used by the backend / offline queue.

    from extraction import extract
    result = extract("photo.jpg")            # local OCR backend (default, offline)
    result = extract("photo.jpg", backend="vlm")   # fine-tuned vision model reads (GPU), offline
    result = extract(jpeg_bytes, backend="claude")
    print(result.model_dump_json(indent=2))
"""

from __future__ import annotations

from .imageproc import ImageError, load_image
from .schema import PageExtraction


class ExtractionError(Exception):
    """Raised when a page cannot be processed. ``code`` tells the queue what to do:

    unreadable_image     -> ask for a new photo (record state: processing failed)
    backend_unavailable  -> keep the record in PENDING_AI and retry later, or manual entry
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


_backends: dict = {}


def get_backend(name: str):
    if name not in _backends:
        if name == "local":
            from .backends.local import LocalBackend

            _backends[name] = LocalBackend()
        elif name == "vlm":
            from .backends.local import LocalBackend

            _backends[name] = LocalBackend(reader="vlm")
        elif name == "claude":
            from .backends.claude import ClaudeBackend

            _backends[name] = ClaudeBackend()
        else:
            raise ValueError(f"unknown backend {name!r} (use 'local', 'vlm' or 'claude')")
    return _backends[name]


def extract(image, backend: str = "local") -> PageExtraction:
    """Photo of one registry page -> structured fields with status and confidence."""
    try:
        img = load_image(image)
    except ImageError as e:
        raise ExtractionError("unreadable_image", str(e)) from e
    try:
        return get_backend(backend).extract(img)
    except ExtractionError:
        raise
    except Exception as e:  # model missing, API down, timeout...
        raise ExtractionError("backend_unavailable", f"{backend}: {e}") from e
