"""Where a workflow gets its pixels, and where it puts them.

Two nodes. Reading goes through :func:`mozo.image.load_image` rather than a decoder of its own,
because channel order is created at decode and invisible afterwards -- a second decoder here would
not be a duplicate but a second answer, and the wrong one fails silently. Writing goes through PIL
for the same reason from the other end: ``cv2.imwrite`` expects BGR, so writing mozo's RGB array
with it would need a flip, and a flip is exactly the thing worth not having to get right twice.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image as PillowImage

from mozo.image import load_image as decode

from ..node import Image
from ..registry import node


@node(category="Input")
def load_image(image: Optional[str] = None) -> Image:
    """Read an image from a path.

    The parameter is called *image* so that running a workflow on something else reads the way it
    should: ``workflow.run(image="street.jpg")``. It is optional because a workflow is commonly
    saved with no path at all and given one per run -- which the catalogue now says, rather than
    leaving an empty string to mean it.
    """
    # Blank as well as unset: a form field sends "" where a Python caller sends None, and both
    # mean the same thing. This is not the sentinel it replaced -- the catalogue says the parameter
    # is optional, and "" is simply another way to have said nothing.
    if not image:
        raise ValueError("no image to load -- set this node's path, or pass run(image=...)")
    return decode(image)


@node(category="Output")
def save_image(image: Image, path: str = "output.jpg") -> None:
    """Write an image to a file."""
    PillowImage.fromarray(image).save(path)
