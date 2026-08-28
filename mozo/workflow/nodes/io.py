"""Where a workflow gets its pixels, and where it puts them.

Three nodes: one source and two sinks.

**One source, not one per kind of file.** A photograph and a two-hour video differ in how many
items they are, and a source already says that -- it yields, so one yield is an image and two
hundred thousand is a video. Nothing else about them differs: same absent inputs, same single
:data:`~mozo.workflow.node.Image` out, same question asked of the person at the editor, which is
"what should this run on". Two nodes made a person answer that question by first classifying their
own file, and then made the answer unreachable anyway -- the editor's file picker was built for
the image node and would not offer an ``.mp4`` to the video one.

The sinks stay two, and by the same test rather than in spite of it.
:func:`save_video` declares ``ordered``, which narrows its stage to one item at a time so frames
are written in the order they were shot. A merged sink would impose that on saving a directory of
images, where there is no order to keep and the narrowing is pure loss.

None of them decode or encode anything themselves. Pixels are PixelFlow's, the way boxes are --
:class:`pixelflow.VideoReader`, :class:`pixelflow.VideoWriter` and :func:`pixelflow.save_image` do
the work, and what these nodes contribute is the declaration: which ports, which parameters, and
what the run is. That is not tidiness. Channel order is created at decode and invisible afterwards,
so a second decoder here would not be a duplicate but a second answer, and the wrong one fails in
silence. One decoder, in the library whose whole contract is that images are RGB.
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Optional

import pixelflow as pf

from mozo.image import load_image as decode

from ..node import Context, Image, Source, State
from ..registry import node, source


#: Extensions read as video. Everything else is decoded as one image.
#:
#: The extension rather than the content, because a person choosing a file already knows which kind
#: it is and the name they gave it is where they said so. Sniffing the bytes would be more clever
#: and less predictable: being wrong would read one frame of a film with nothing to indicate why.
#: This belongs in PixelFlow eventually, next to the decoders it selects between.
VIDEO_SUFFIXES = frozenset({
    ".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".wmv", ".flv", ".ts",
})


@source(category="Input")
def read_media(run: Context, source: Optional[Source] = None, stride: int = 1,
               start: int = 0, count: Optional[int] = None) -> Image:
    """Read an image or a video: one frame for the first, every frame for the second.

    Every workflow starts here. It is a source, so it is asked once for a sequence and what it
    yields is the run -- one item for a photograph, two hundred thousand for a film, and the same
    graph downstream of both. That is what lets a workflow built on a still image be pointed at
    footage without being rewired, and it is why one node can do both: a source that yields once
    and a source that yields for an hour differ in nothing a graph can observe.

    **The rate it declares is the rate it yields**, because PixelFlow divides by the stride before
    reporting it. Every fifth frame of a 25 fps file is a 5 fps sequence, and a sink writing it
    back at 25 would play five times too fast. Neither this node nor the sink does that arithmetic,
    so neither can get it wrong. A still image has no rate at all and declares none, which is why
    :func:`save_video` behind one asks to be told.

    Args:
        source: The file to read. :data:`~mozo.workflow.node.Source` rather than ``str`` for the
            same reason ``Color`` is not ``str``: it is a path either way, but it is the one
            parameter a person at a browser cannot write down, since their file is on their machine
            and the path would name the server's. Saying so is what puts a file picker on the node
            instead of a box nobody can fill in. Optional because a workflow is commonly saved with
            no file and given one per run.
        stride: Take every *stride*-th frame. PixelFlow walks past the others without decoding
            them -- measured at 2.43x on 720p -- and divides the declared rate to match. Ignored
            for an image, which is one frame however you step through it.
        start: Skip this many frames first.
        count: Stop after this many frames. Unset reads to the end.
    """
    # Blank as well as unset: a form field sends "" where a Python caller sends None, and both mean
    # the same thing. Asked of the two things that can be blank rather than of the value, because
    # ``not source`` reads an array as a truth value, which numpy refuses for anything but a single
    # element -- so every frame handed to ``run_many``, which its own docstring says it takes,
    # raised here instead of being decoded.
    if source is None or (isinstance(source, (str, bytes, bytearray)) and not source):
        raise ValueError(
            "nothing to read -- choose a file on this node, or pass run(source=...)")

    named = isinstance(source, (str, Path))
    if named and Path(source).suffix.lower() in VIDEO_SUFFIXES:
        reader = pf.VideoReader(str(source), stride=stride, start=start)
        try:
            run.declare(name=str(source), fps=reader.fps, width=reader.width,
                        height=reader.height, frames=reader.frames, is_live=reader.is_live)
            yield from (reader if count is None else islice(reader, count))
        finally:
            reader.close()
    else:
        # Bytes or an array is already-decoded pixels, so one image with no name of its own.
        frame = decode(source)
        height, width = frame.shape[:2]
        # Declared before the yield like a video's, and for the same reason: a sink downstream
        # opens itself from these and must not have to know which kind of file was upstream.
        # ``fps`` is None because a photograph has no rate -- which is what makes a video sink ask.
        run.declare(name=str(source) if named else "an image", fps=None,
                    width=width, height=height, frames=1, is_live=False)
        yield frame


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
