"""
Command-line interface for Mozo.

Provides simple commands for starting the server and checking version.
"""

import click
import uvicorn


@click.group()
def cli():
    """Mozo - Universal CV Model Server"""
    pass


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, type=int, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload on code changes')
@click.option('--workers', default=1, type=int, help='Number of worker processes')
def start(host, port, reload, workers):
    """Start the Mozo model server"""
    click.echo(f"Starting Mozo server on {host}:{port}...")
    if reload:
        click.echo("Auto-reload enabled (development mode)")
    if workers > 1:
        click.echo(f"Running with {workers} worker processes")

    uvicorn.run(
        "mozo.server:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1  # reload only works with 1 worker
    )


@cli.command()
@click.argument("workflow", type=click.Path(exists=True, dir_okay=False))
@click.option("--image", help="Run on this image instead of the path saved in the workflow")
@click.option("--set", "settings", multiple=True, metavar="NAME=VALUE",
              help="Override a parameter. Repeatable.")
def run(workflow, image, settings):
    """Run a workflow from a JSON file, with no browser and no server.

    Prints one line per node: what it produced, in words. An image says its size, detections say
    how many and what they were, a depth map says its range -- because a terminal is not a place
    to put a photograph, and the thing worth seeing headlessly is whether the graph did its job.
    Nodes that write files have already written them.
    """
    from mozo.workflow import Workflow

    overrides = {}
    for setting in settings:
        name, separator, value = setting.partition("=")
        if not separator:
            raise click.BadParameter(f"expected NAME=VALUE, got {setting!r}")
        overrides[name] = value
    if image:
        overrides["image"] = image

    built = Workflow.load(workflow)
    for event in built.stream(**overrides):
        if event.status == "failed":
            raise click.ClickException(event.error)
        if event.status == "completed":
            ends = " (an end)" if event.node in built.terminals else ""
            click.echo(f"  {event.node}: {_describe(event.output)}{ends}")


def _describe(value) -> str:
    """What a node produced, in one line."""
    import numpy as np

    if value is None:
        return "nothing"
    if isinstance(value, tuple):
        return ", ".join(_describe(part) for part in value)
    if isinstance(value, list):
        return f"{len(value)} x [{_describe(value[0])}]" if value else "an empty batch"
    if isinstance(value, np.ndarray):
        if value.ndim == 2:
            return f"a {value.shape[1]}x{value.shape[0]} map, {value.min():.3g} to {value.max():.3g}"
        return f"a {value.shape[1]}x{value.shape[0]} image"
    named = getattr(value, "to_dict", None)
    if named:
        rows = named()
        seen = {row.get("class_name") for row in rows if isinstance(row, dict)}
        return f"{len(rows)} x {sorted(name for name in seen if name)}" if rows else "nothing found"
    return type(value).__name__


@cli.command()
def version():
    """Show Mozo version"""
    from mozo import __version__
    click.echo(f"Mozo version {__version__}")


if __name__ == '__main__':
    cli()
