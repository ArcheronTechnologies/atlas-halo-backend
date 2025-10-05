"""
Prediction Validation & Refinement Scheduler
Runs daily validation and refinement tasks
"""

import asyncio
from datetime import datetime, time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PredictionScheduler:
    """Schedules and manages prediction validation and refinement tasks"""

    def __init__(self, db, model_path: str):
        self.db = db
        self.model_path = model_path
        self.is_running = False
        self.task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the scheduler"""
        if self.is_running:
            logger.warning("Scheduler already running")
            return

        self.is_running = True
        self.task = asyncio.create_task(self._run_scheduler())
        logger.info("✅ Prediction scheduler started")

    async def stop(self):
        """Stop the scheduler"""
        self.is_running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Prediction scheduler stopped")

    async def _run_scheduler(self):
        """Main scheduler loop - runs validation and refinement daily"""
        while self.is_running:
            try:
                # Calculate seconds until next 3 AM
                now = datetime.now()
                target_time = time(3, 0)  # 3:00 AM

                # Create datetime for next 3 AM
                next_run = datetime.combine(now.date(), target_time)
                if now.time() > target_time:
                    # If it's already past 3 AM today, schedule for tomorrow
                    from datetime import timedelta
                    next_run += timedelta(days=1)

                seconds_until_run = (next_run - now).total_seconds()

                logger.info(f"⏰ Next validation/refinement scheduled for {next_run.isoformat()}")
                logger.info(f"   (in {seconds_until_run / 3600:.1f} hours)")

                # Wait until next run time
                await asyncio.sleep(seconds_until_run)

                # Run the daily tasks
                await self._run_daily_tasks()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Scheduler error: {e}")
                # Wait 1 hour before retrying if there's an error
                await asyncio.sleep(3600)

    async def _run_daily_tasks(self):
        """Execute daily validation and refinement"""
        logger.info("=" * 80)
        logger.info(f"🔄 Starting daily prediction tasks at {datetime.now().isoformat()}")
        logger.info("=" * 80)

        try:
            # Step 1: Validate predictions from last 24 hours
            logger.info("📊 Step 1: Validating predictions...")
            from backend.ml_training.prediction_validator import run_daily_validation
            validation_results = await run_daily_validation(self.db)

            if validation_results['status'] == 'success':
                logger.info(f"✅ Validation complete:")
                logger.info(f"   - Total predictions: {validation_results['total_predictions_validated']}")
                logger.info(f"   - Average accuracy: {validation_results['average_accuracy']:.1%}")
                logger.info(f"   - Models tested: {', '.join(validation_results['model_versions_tested'])}")

            # Step 2: Apply refinements based on validation feedback
            logger.info("🔧 Step 2: Applying refinements...")
            from backend.ml_training.prediction_refiner import PredictionRefiner
            refiner = PredictionRefiner(self.db)
            await refiner.apply_feedback_refinement(validation_results)

            # Get refinement summary
            summary = await refiner.get_adjustment_summary()
            if summary.get('summary'):
                logger.info(f"✅ Refinement complete:")
                logger.info(f"   - Locations adjusted: {summary['summary']['total_locations']}")
                logger.info(f"   - Over-predicted locations: {summary['summary']['locations_over_predicted']}")
                logger.info(f"   - Under-predicted locations: {summary['summary']['locations_under_predicted']}")
                logger.info(f"   - Well-calibrated locations: {summary['summary']['locations_well_calibrated']}")

            # Step 3: Generate new refined predictions for next 24 hours
            logger.info("🔮 Step 3: Generating new refined predictions...")
            total_new = await refiner.generate_refined_predictions(self.model_path)
            logger.info(f"✅ Generated {total_new} new refined predictions")

            # Step 4: Clean up old expired predictions
            logger.info("🧹 Step 4: Cleaning up old predictions...")
            cleanup_query = """
            DELETE FROM hotspot_predictions
            WHERE valid_until < NOW() - INTERVAL '7 days'
            """
            result = await self.db.execute_query(cleanup_query)
            logger.info(f"✅ Cleaned up old predictions")

            logger.info("=" * 80)
            logger.info(f"✅ Daily prediction tasks completed successfully at {datetime.now().isoformat()}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"❌ Daily tasks failed: {e}", exc_info=True)

    async def run_now(self):
        """Manually trigger validation and refinement immediately"""
        logger.info("🔄 Manual trigger: Running validation and refinement now...")
        await self._run_daily_tasks()


# Singleton instance
_scheduler_instance: Optional[PredictionScheduler] = None


async def get_prediction_scheduler(db, model_path: str) -> PredictionScheduler:
    """Get or create the scheduler singleton"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = PredictionScheduler(db, model_path)
    return _scheduler_instance


async def start_prediction_scheduler(db, model_path: str):
    """Start the prediction scheduler service"""
    scheduler = await get_prediction_scheduler(db, model_path)
    await scheduler.start()


async def stop_prediction_scheduler():
    """Stop the prediction scheduler service"""
    global _scheduler_instance
    if _scheduler_instance:
        await _scheduler_instance.stop()
        _scheduler_instance = None