"""Where a workflow gets its pixels, and where it puts them.

Four nodes: one source and three sinks.

**One source, not one per kind of file.** A photograph and a two-hour video differ in how many
items they are, and a source already says that -- it yields, so one yield is an image and two
hundred thousand is a video. Nothing else about them differs: same absent inputs, same single
:data:`~mozo.workflow.node.Image` out, same question asked of the person at the editor, which is
"what should this run on". Two nodes made a person answer that question by first classifying their
own file, and then made the answer unreachable anyway -- the editor's file picker was built for
the image node and would not offer an ``.mp4`` to the video one.

The two image sinks stay two, and by the same test rather than in spite of it.
:func:`save_video` declares ``ordered``, which narrows its stage to one item at a time so frames
are written in the order they were shot. A merged sink would impose that on saving a directory of
images, where there is no order to keep and the narrowing is pure loss.

:func:`save_annotations` is the third, and the only one here that writes something other than
pixels. It is a sink by the same definition as the others -- inputs, no output, a file at the end
of it -- and it lives beside them because that is what this module is: where a workflow's values
leave the process. A second annotation sink would earn its own module; one does not.

It delegates like the rest: a detection becomes JSON by PixelFlow's ``to_dict``, the way pixels
become a file by ``save_image``. :mod:`mozo.workflow.wire` does the same for a detection on its way
to a browser and keeps every key; this drops the empty ones. Two products, deliberately -- a
transport payload is read once and discarded, a dataset line is read a million times -- and both
say so where they diverge.

None of them decode or encode anything themselves. Pixels are PixelFlow's, the way boxes are --
:class:`pixelflow.VideoReader`, :class:`pixelflow.VideoWriter` and :func:`pixelflow.save_image` do
the work, and what these nodes contribute is the declaration: which ports, which parameters, and
what the run is. That is not tidiness. Channel order is created at decode and invisible afterwards,
so a second decoder here would not be a duplicate but a second answer, and the wrong one fails in
silence. One decoder, in the library whose whole contract is that images are RGB.
"""

from __future__ import annotations

import json
import os
import re
from itertools import islice
from pathlib import Path
from typing import Optional

import pixelflow as pf

from mozo.image import load_image as decode

from ..node import Context, Detections, Image, Source, State
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

#: What counts as an image when reading a folder. Only used there: a file named on its own is
#: decoded whatever it is called, because naming one is asking for it. In a folder the same list
#: is a filter, since a folder of photographs also holds ``.DS_Store`` and a README, and refusing
#: to run because of those would be refusing the ordinary case.
IMAGE_SUFFIXES = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".ppm", ".pgm",
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

    path = Path(source) if isinstance(source, (str, Path)) else None

    if path is not None and path.is_dir():
        # Truncated before anything is declared, so what it says it will yield is what it yields.
        # ``None`` slices to the whole list, which is why there is no second branch here -- and
        # the video branch below needs one only because a reader has no slice.
        found = _listing(path)[:count]
        if not found:
            raise ValueError(
                f"no images in {source} -- looked for {', '.join(sorted(IMAGE_SUFFIXES))}")
        run.declare(name=str(path), frames=len(found),
                    labels=tuple(item.stem for item in found))
        for item in found:
            yield decode(item)
        return

    if path is not None and path.suffix.lower() in VIDEO_SUFFIXES:
        reader = pf.VideoReader(str(path), stride=stride, start=start)
        try:
            # What it will yield, not what the file holds. PixelFlow already divides its count by
            # the stride; ``count`` is this node's own limit, so correcting for it is this node's
            # job -- and a sink deciding whether it can take one filename reads this number.
            # ``filter(None, ...)`` drops whatever nobody knows, which for a rate-less reader is
            # its count, and None is the only honest answer to how many are coming.
            run.declare(name=str(path), fps=reader.fps, width=reader.width,
                        height=reader.height, is_live=reader.is_live,
                        frames=min(filter(None, (reader.frames, count)), default=None))
            yield from islice(reader, count)
        finally:
            reader.close()
        return

    # Bytes or an array is already-decoded pixels, so one image with no name of its own.
    frame = decode(source)
    height, width = frame.shape[:2]
    # Declared before the yield like the others, and for the same reason: a sink downstream opens
    # itself from these and must not have to know which kind of source was upstream. ``fps`` is
    # None because a photograph has no rate -- which is what makes a video sink ask for one.
    run.declare(name=str(path) if path else "an image", width=width, height=height, frames=1,
                # Its own name, so one photograph through a folder sink comes back as itself
                # rather than as itself with an index glued on. Bytes have no name to keep.
                labels=(path.stem,) if path else None)
    yield frame


#: What can be empty. A tuple of names inside the function body is six ``LOAD_GLOBAL``s and a
#: ``BUILD_TUPLE`` per call, and this is called once per key per detection -- 221 million times
#: over a million images. Measured, 179 ns a call against 154 hoisted.
_EMPTY = (str, bytes, list, tuple, dict, set)


def _nothing(value) -> bool:
    """Is *value* the absence of a value, rather than a value?

    What :func:`save_annotations` leaves out of a line. None and an empty collection both mean
    nothing was there, and so does the key not being present -- so dropping them is lossless, and
    on a real detection it is 151 bytes against 98. Zero is not among them: a confidence or a
    duration of zero is a measurement, and a reader cannot tell one this dropped from one that was
    never taken.

    A rule about values, deliberately, and not a list of keys to skip. A list would be a second
    statement of what a detection is, next to PixelFlow's, and the two would drift apart the first
    time a field was added there.

    Asked of the container types by name rather than by comparing against ``[]`` and ``{}``.
    ``value == []`` is an array comparison when the value is a numpy anything, which returns an
    empty array, which ``or`` then raises on -- the same shape of break that ``not source`` was in
    :func:`read_media`. :meth:`~pixelflow.Detections.to_dict` converts numpy out before returning,
    so nothing here reaches that today; this does not depend on it continuing to.
    """
    return value is None or (isinstance(value, _EMPTY) and not value)


def _listing(folder: Path) -> list:
    """The images in *folder*, in the order a person would put them in.

    ``os.scandir`` rather than ``iterdir`` or ``glob``: those stat every entry to answer
    ``is_file()``, where scandir reads the kind from the directory entry it already has. Measured
    on 10,000 files, 7.8 ms against 77.7 ms.

    **Sorted by digit runs, not by character.** Plain sorting puts ``frame_10`` before ``frame_2``,
    which turns a frame sequence into a shuffled one -- and a shuffled sequence written back out as
    a video plays, which is what makes it the wrong kind of mistake. On zero-padded names the two
    orders agree, so this costs nothing and only ever fixes.

    Top level only. Recursing would quietly pick up a thumbnails folder sitting beside the photos,
    and a run that processed more than you pointed at is worse than one that processed less.
    """
    found = [Path(entry.path) for entry in os.scandir(folder)
             if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_SUFFIXES]
    return sorted(found, key=lambda path: [int(part) if part.isdigit() else part.lower()
                                           for part in re.split(r"(\d+)", path.name)])


@node(category="Output")
def save_image(image: Image, run: Context, path: str = "output",
               format: str = ".jpg") -> None:
    """Write each image to its own file.

    **A folder writes one file per item, named for the item.** So a folder of photographs comes
    back as a folder of the same names, and that is the whole of it -- ``photos/cat.jpg`` in,
    ``out/cat.jpg`` out. :attr:`~mozo.workflow.node.Context.label` is where the name comes from,
    and every source has one, so this does not ask which kind of run it is in.

    It used to take one filename and write it once per item. Ten frames in produced one file,
    overwritten nine times, with nothing raised -- the same shape of wrong as a video that plays
    at the wrong speed. Now a filename is refused as soon as the run *says* it has more than one
    item, which is before anything is written and before the second item exists.

    Refused in a preview as well, where only one item is taken and one filename would have been
    fine. A preview that worked where the run it previews would fail is a preview that told you
    the wrong thing, and the wiring it is reporting on is the same wiring.

    Args:
        path: A folder, made if it is not there. A name with a suffix instead writes that one file,
            which is right for a run of one item and refused for a run of many.
        format: The suffix to write. Stated rather than carried over from the input, because the
            output is not the input file -- it has been decoded, run through the graph and encoded
            again, so the container the bytes arrived in is not a property they still have.
    """
    target = Path(path)
    if not target.suffix:
        target.mkdir(parents=True, exist_ok=True)
        target = target / f"{run.label}.{format.lstrip('.')}"
    elif run.is_live or (run.frames or 1) > 1:
        # ``is_live`` as well as the count, because a camera declares no count at all -- and
        # ``frames`` is documented as an estimate and the wrong thing to decide with, so a source
        # that cannot count itself would otherwise be the one place the overwriting survived.
        raise ValueError(
            f"{run.frames or 'unboundedly many'} images to write but one filename, {path!r}. "
            f"Give a folder and each is written under its own name.")
    pf.save_image(str(target), image)


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


@node(category="Output", ordered=True)
def save_annotations(image: Image, detections: Detections, run: Context, state: State,
                     path: str = "annotations.jsonl") -> None:
    """Write what was detected, one line of JSON per item, as each item finishes.

    **Written as it happens, not gathered and written at the end.** That is the whole design, and
    it is the difference between a run that survives being interrupted and one that does not.
    Gathering a million items to write one document at the end costs 11.6 GB of Python objects
    before any file exists -- so it does not merely lose everything to a crash at item 500,000, it
    runs out of memory long before reaching one. Writing one costs 0.037 ms measured end to end,
    of which the append and its flush are 0.003 -- against a single SAM 3 inference, about a
    five-hundredth. There is no throughput argument for the other way; it is slower *and* it loses
    the run.

    So a cancelled run leaves a complete file of everything that finished, the same way a cancelled
    :func:`save_video` leaves a video that plays. :meth:`~mozo.workflow.node.State.on_close` is
    what closes the handle, and it runs however the run ended.

    **JSONL rather than a dataset format, and deliberately not usable as-is.** COCO, YOLO and CSV
    are each one document with a shape -- a global category table, boxes normalised by an image
    size, one row per detection rather than per image -- and none of them can be appended to. What
    can be appended to is a line at a time, so that is what is written, and converting it is a
    reader's job. The conversion is also not once: the same run feeds a labelling tool as COCO, a
    trainer as YOLO and a report as CSV, and a format chosen here would force one of them and make
    the other two a re-run. Nothing converts yet. This is the file those converters will read.

    **The size comes from the image rather than from the run.** Not only because a folder source
    declares none -- every photograph in it differs -- but because the size on the line has to be
    the one the boxes were measured against. Behind a ``resize`` the run's own figures are the
    source's and would be confidently wrong, where the array is right in every graph. That is why
    this is an input and not a fact :class:`~mozo.workflow.node.Context` should learn to carry.

    **A line has to carry what nothing else remembers.** Which is the test for what belongs on it:
    ``label`` is gone the moment the run ends, ``time`` needs a rate only the source knew, and the
    detections are the point. ``width`` and ``height`` could be read back off the images, but
    needing the images to convert the annotations is exactly the coupling this avoids -- and both
    COCO and YOLO ask for them. Everything else is derivable and left out.

    One is deferred rather than decided: :attr:`~mozo.workflow.node.Context.name`, the folder or
    file the run read. It is a constant, so carrying it is the same forty bytes on every one of a
    million lines, and JSONL has no header to put it in instead. A converter needing it can be told
    where the images are, which it has to be anyway to resolve a stem to a file. Adding a field
    later is backwards compatible; discovering one is missing after a million inferences is not.

    **An image with nothing in it still gets a line.** A dataset needs its negatives, and no line
    is indistinguishable from not processed.

    The detections themselves are :meth:`pixelflow.Detections.to_dict`'s, minus the keys holding
    nothing -- 338 bytes a detection against 98, measured on a real RF-DETR result. Not a field
    list of this module's own: what a detection is, is PixelFlow's to say, and a second answer here
    is one that goes stale. It already PNG-encodes masks to base64, so a SAM 3 mask costs 1.5 KB
    rather than the four megabytes of nested booleans a raw dump would be.

    ``ordered`` because one open file has one writer. It is not for the order -- though the lines
    come out in item order, which is what lets a reader trust the file's shape -- it is that an
    ``exclusive`` node is sized by ``model_workers``, so a caller who raised that for their model
    would otherwise put four threads inside this one handle.

    Args:
        path: The file to write. Its folder is made if it is not there. Truncated at the start of
            each run rather than appended to: two runs sharing a path would interleave two datasets
            under one set of indices, which reads as one dataset and is not.
    """
    handle = state.get("handle")
    if handle is None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = open(target, "w", encoding="utf-8")
        state.on_close(handle.close)
        state["handle"] = handle

    height, width = image.shape[:2]
    when = run.time
    record = {"index": run.index, "label": run.label, "width": width, "height": height,
              "detections": [{key: value for key, value in found.items() if not _nothing(value)}
                             for found in detections.to_dict()]}
    if when is not None:
        record["time"] = when
    # One write of the whole line, flushed. The flush is what makes the claim above true: without
    # it the last few kilobytes live in a buffer that a killed process never gets to empty.
    handle.write(json.dumps(record) + "\n")
    handle.flush()
