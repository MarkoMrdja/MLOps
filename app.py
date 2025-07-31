from api.api import app
from utils.logger import logger

def run_flask():
    """
    Run the Flask API.
    """
    logger.info("Starting Flask application on host 0.0.0.0, port 5000")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        logger.error(f"Failed to start Flask app: {str(e)}")
        raise

if __name__ == '__main__':
    logger.info('Starting application...')
    run_flask()