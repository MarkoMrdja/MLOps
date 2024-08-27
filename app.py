import multiprocessing
import subprocess
from api.api import app

def run_flask():
    """
    Run the Flask API.
    """
    app.run(debug=True)


if __name__ == '__main__':
    run_flask()