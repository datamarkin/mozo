import uvicorn

def main():
    """
    This is the main entry point when running `python -m mozo`.
    It launches the Uvicorn server.
    """
    print("Starting Mozo server via `python -m mozo`...")
    # Uvicorn will look for the 'app' instance in the 'mozo.server' module.
    uvicorn.run("mozo.server:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
