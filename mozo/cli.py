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
def version():
    """Show Mozo version"""
    from mozo import __version__
    click.echo(f"Mozo version {__version__}")


if __name__ == '__main__':
    cli()
