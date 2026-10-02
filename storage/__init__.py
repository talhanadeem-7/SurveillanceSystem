class StorageFatalError(RuntimeError):
    """Persistence failed; stop surveillance rather than skip a video frame."""
