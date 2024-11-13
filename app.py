import multiprocessing
from api.api import app
from metaflow import namespace
from pipelines.fashion_mnist_flow import FashionMNISTFlow
import logging
from apscheduler.schedulers.background import BackgroundScheduler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline():
    """
    Run the Metaflow pipeline.
    """
    try:
        logger.info("Starting pipeline run")
        namespace('prod')
        FashionMNISTFlow()
        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

def run_scheduler():
    """
    Run the scheduler for the pipeline.
    """
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        func=run_pipeline,
        trigger='cron',
        hour=2,  # Run at 2 AM
        id='training_pipeline'
    )
    scheduler.start()
    logger.info("Scheduler started - Pipeline will run daily at 2 AM")
    
    try:
        # Keep the scheduler thread alive
        while True:
            pass
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()

def run_flask():
    """
    Run the Flask API.
    """
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    api_process = multiprocessing.Process(target=run_flask)
    scheduler_process = multiprocessing.Process(target=run_scheduler)

    # Start both processes
    api_process.start()
    scheduler_process.start()

    try:
        api_process.join()
        scheduler_process.join()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        api_process.terminate()
        scheduler_process.terminate()
        api_process.join()
        scheduler_process.join()