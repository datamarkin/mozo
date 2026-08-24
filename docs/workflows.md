# Workflows

A model on its own is rarely the job. Detect, then blur the faces, then save — that is a workflow.

mozo runs them as a graph of nodes in a JSON file. Draw it in the browser at `/workflow`, keep it in
version control, and run it from Python, the command line, or over HTTP. Same models, same process,
nothing extra to install.

## Running one

```python
from mozo.workflow import Workflow

workflow = Workflow.load("blur_faces.json")
results = workflow.run(image="street.jpg")

results[workflow.terminals[0]]      # what came out of the end
```

```bash
mozo run blur_faces.json --image street.jpg
```

`run()` gives back every node's output keyed by node id — an intermediate result is often the one
worth looking at, and hiding it would mean running the whole thing again to see it. `stream()`
reports each node as it starts and finishes, which is what the editor draws.

Overrides address a parameter by name: `run(image=..., threshold=0.6)`. A name more than one node
uses is refused rather than guessed at, and the message names the nodes that have it.

## What a workflow is

```json
{
  "nodes": [
    {"id": "load-1", "type": "load_image",
     "position": {"x": 0, "y": 0},
     "data": {"parameters": {"image": "street.jpg"}}},
    {"id": "detect-1", "type": "yolov26",
     "data": {"parameters": {"variant": "seg-nano", "threshold": 0.5}}}
  ],
  "edges": [
    {"source": "load-1", "sourceHandle": "image",
     "target": "detect-1", "targetHandle": "image"}
  ]
}
```

A workflow either is valid or does not exist. Loading one checks that every connection names ports
that exist, that the types on both ends agree, that every input is fed exactly once, and that the
graph is acyclic — so a workflow that loaded will not fail for a structural reason halfway through a
run with a model already on the GPU.

## Nodes

A node is an ordinary Python function. Its signature is its declaration:

```python
@node(category="Annotate")
def draw_boxes(image: Image, detections: Detections,
               thickness: int | None = None, color: Color | None = None) -> Image:
    """Draw a box around each detection."""
    return pf.annotate.box(image.copy(), detections, thickness=thickness, colors=_colors(color))
```

One rule splits the two kinds of argument: **an annotation that names a port type is an input;
anything else is a parameter.** Inputs are wired from other nodes; parameters are typed in. The
widget the editor draws comes from the annotation too — `int`, `float`, `str`, `bool`, `Color`, and
`Literal[...]` for a choice. `int | None` means the parameter may be left unset, which is how a
thickness scales itself to the image.

The catalogue the editor reads is generated from these declarations, so it cannot drift from what
runs. Adding a node is adding a function.

## Port types

A connection is only allowed between ports of the same type.

| | |
|---|---|
| `IMAGE` | `HxWx3` RGB `uint8` — mozo's image contract |
| `DETECTIONS` | PixelFlow `Detections`: boxes, masks, keypoints, read text |
| `CLASSIFICATIONS` | PixelFlow `Classifications` — scores with no boxes |
| `DEPTH` | a float depth map with a range |
| `EMBEDDING` | an `NxD` float array |

`CLASSIFICATIONS` is separate from `DETECTIONS` although both are PixelFlow types, because
`Classifications` has no boxes at all — sharing a port type would let the editor offer a connection
that fails the moment it runs. `DEPTH` is separate from `IMAGE` because a depth map is a
measurement: flattened to eight bits it is a picture, and the metric Depth Anything variants lose
the only thing that distinguishes them from the relative ones.

## Batching

A node written for one image runs over fifty without knowing. A list arriving on an input fans the
node out and collects the results, and that lives in the engine so every node gets it from one
implementation. `crop_around_detections` turns one image into one per detection, which every node
downstream then runs over.

Lists of different lengths are refused rather than padded.

## Over HTTP

```
GET  /workflow            the editor
GET  /workflow/nodes      the catalogue
POST /workflow/validate   build it, and say why not
POST /workflow/run        run it
POST /workflow/stream     run it, one server-sent event per node
```

`run` and `stream` take the document as a form field, an optional uploaded image, and `include`,
which is `terminals` by default. Sending every node's output means encoding every intermediate: on
one 1920×1281 photograph a five-node workflow takes 27 ms to run and 347 ms to encode. The editor
asks for `all`; a batch job over full-resolution frames should not have to.

## What is not there

- **Tracking.** Stateful across frames, and mozo has no concept of a video.
- **Zones**, and **SAM 2** / **EdgeTAM**. All three are prompted with pixel coordinates, and the
  editor has no way yet to express a click. Everything else about those models already works
  through `mozo.get_model`.
