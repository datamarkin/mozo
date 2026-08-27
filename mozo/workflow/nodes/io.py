"""Where a workflow gets its pixels, and where it puts them.

Four nodes: two sources and two sinks, one of each for a single image and for a video.

None of them decode or encode anything themselves. Pixels are PixelFlow's, the way boxes are --
:class:`pixelflow.VideoReader`, :class:`pixelflow.VideoWriter` and :func:`pixelflow.save_image` do
the work, and what these nodes contribute is the declaration: which ports, which parameters, and
what the run is. That is not tidiness. Channel order is created at decode and invisible afterwards,
so a second decoder here would not be a duplicate but a second answer, and the wrong one fails in
silence. One decoder, in the library whose whole contract is that images are RGB.
"""

from __future__ import annotations

from itertools import islice
from typing import Optional

import pixelflow as pf

from mozo.image import load_image as decode

from ..node import Context, Image, Source, State
from ..registry import node, source


@node(category="Input")
def load_image(image: Optional[Source] = None) -> Image:
    """Read an image from a path.

    The parameter is called *image* so that running a workflow on something else reads the way it
    should: ``workflow.run(image="street.jpg")``. It is optional because a workflow is commonly
    saved with no path at all and given one per run -- which the catalogue now says, rather than
    leaving an empty string to mean it.

    :data:`~mozo.workflow.node.Source` rather than ``str`` for the same reason ``Color`` is not
    ``str``: it is a path either way, but it is the one parameter whose value a person at a browser
    has no way to write down, since their file is on their machine and the path would have to name
    the server's. Saying so in the annotation is what puts a file picker on the node instead of a
    text box nobody can fill in.
    """
    # Blank as well as unset: a form field sends "" where a Python caller sends None, and both
    # mean the same thing. This is not the sentinel it replaced -- the catalogue says the parameter
    # is optional, and "" is simply another way to have said nothing.
    #
    # Asked of the two things that can be blank rather than of the value: ``not image`` reads an
    # array as a truth value, which numpy refuses for anything but a single element, so every frame
    # handed to ``run_many`` -- which its own docstring says it takes -- raised here instead of
    # being decoded. Empty bytes are the same claim as an empty string and are refused with it.
    if image is None or (isinstance(image, (str, bytes, bytearray)) and not image):
        raise ValueError("no image to load -- set this node's path, or pass run(image=...)")
    return decode(image)


@source(category="Input")
def read_video(run: Context, path: Optional[Source] = None, stride: int = 1,
               start: int = 0, count: Optional[int] = None) -> Image:
    """Read a video file, one frame at a time.

    A source rather than an ordinary node, because a video is one node and a great many items: the
    run is a pass over what this yields, and it yields rather than returning, which is why a
    two-hour file costs what a ten-second one costs.

    The decoding is PixelFlow's, for the reason every other node's work is:
    :class:`pixelflow.VideoReader` owns frames the way ``pf.annotate`` owns boxes, and a second
    decoder here would be a second answer rather than a duplicate. What this node contributes is
    the declaration -- which parameters, and what the run is.

    **The rate it declares is the rate it yields**, because PixelFlow divides by the stride before
    reporting it. Every fifth frame of a 25 fps file is a 5 fps sequence, and a sink writing it
    back at 25 would play five times too fast. Neither this node nor the sink does that arithmetic,
    so neither can get it wrong.

    Args:
        path: The file to read. Optional for the same reason :func:`load_image`'s is -- a workflow
            is commonly saved with no file and given one per run.
        stride: Take every *stride*-th frame. PixelFlow walks past the others without decoding
            them, and divides the declared rate to match.
        start: Skip this many frames first.
        count: Stop after this many frames. Unset reads to the end.
    """
    if not path:
        raise ValueError("no video to read -- set this node's path, or pass path=...")
    reader = pf.VideoReader(str(path), stride=stride, start=start)
    try:
        run.declare(name=str(path), fps=reader.fps, width=reader.width,
                    height=reader.height, frames=reader.frames, is_live=reader.is_live)
        yield from (reader if count is None else islice(reader, count))
    finally:
        reader.close()


@node(category="Output")
def save_image(image: Image, path: str = "output.jpg") -> None:
    """Write an image to a file."""
    pf.save_image(path, image)


@node(category="Output", ordered=True)
def save_video(image: Image, run: Context, state: State, path: str = "output.mp4",
               fps: Optional[float] = None) -> None:
    """Write every frame that reaches this node to one video file.

    The first node in mozo that is not one call. A file has a beginning and an end, so this one
    holds an open writer for the length of the run: :class:`~mozo.workflow.node.State` is where it
    keeps it, and ``ordered=True`` is what makes the order it is called in the order the frames are
    in. Both already existed -- ``ordered`` since the pipeline was written, with a video writer as
    the example in its own docstring -- and this is the first node to use either.

    ``ordered`` is not a precaution. Without it the stage widens to *workers* threads and frames
    arrive in whatever order the stage in front finished them, which for a model stage is not the
    order they went in. Measured: 120 frames in, 103 out, every one misplaced -- and the file
    plays, which is the problem.

    The writer is PixelFlow's, and it is handed the run itself: ``like=`` takes a rate from
    anything that reports one, and :class:`~mozo.workflow.node.Context` reports the source's --
    already divided by any stride. So the rate crosses from the source to the sink without being
    written down in between, which is the only place it could have been written down wrong.

    Args:
        path: Where to write.
        fps: The rate to write, where the run's own is wrong or absent. Left unset it comes from
            the source. A run whose source declares no rate -- a live camera has timestamps rather
            than a frame rate -- must be given one here, and PixelFlow says so rather than guessing.
    """
    writer = state.get("writer")
    if writer is None:
        # PixelFlow refuses a rate that is missing, zero or not a number, and refuses a frame whose
        # size differs from the first -- which cv2 would drop in silence, leaving a short file and
        # no sign of why. Neither check is repeated here.
        writer = (pf.VideoWriter(path, fps=fps) if fps is not None
                  else pf.VideoWriter(path, like=run))
        state.on_close(writer.close)
        state["writer"] = writer
    writer.write(image)
