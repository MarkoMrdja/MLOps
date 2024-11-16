from api.api import app
from utils.logger import logger

def run_flask():
    """
    Run the Flask API.
    """
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    logger.info('Starting application...')
    run_flask()