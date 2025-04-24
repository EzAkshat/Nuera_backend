from datetime import datetime
import pytz
import asyncio
import logging
from ai_service import get_ai_response
from database import reminders_collection
from websocket_manager import broadcast_reminder
from bson import ObjectId

logger = logging.getLogger(__name__)

# Always use Asia/Kolkata timezone
KOLKATA_TZ = pytz.timezone("Asia/Kolkata")

async def schedule_reminder_for(reminder_id: str, user_id: str, text: str, when: datetime):
    try:
        # Localize `when` to Kolkata time
        if when.tzinfo is None:
            when_local = KOLKATA_TZ.localize(when)
        else:
            when_local = when.astimezone(KOLKATA_TZ)

        # Current time in Kolkata
        now_local = datetime.now(KOLKATA_TZ)
        delay = (when_local - now_local).total_seconds()

        logger.info(f"[DEBUG] Reminder {reminder_id} when={when_local}, now={now_local}, delay={delay:.1f}s")

        # Sleep in chunks until it's time
        if delay > 0:
            remaining = delay
            while remaining > 0:
                chunk = min(30, remaining)
                await asyncio.sleep(chunk)
                remaining -= chunk
            logger.info(f"Time reached for reminder {reminder_id} (Kolkata time)")
        else:
            logger.info(f"Reminder {reminder_id} is overdue by {-delay:.1f}s; sending immediately")

        # Check reminder not already completed
        reminder = await reminders_collection.find_one({"_id": ObjectId(reminder_id)})
        if not reminder or reminder.get("completed"):
            logger.info(f"Reminder {reminder_id} already done; skipping")
            return

        # Generate AI follow-up
        ai_response = await get_ai_response(text, is_reminder=True)

        # Mark as completed in DB
        await reminders_collection.update_one(
            {"_id": ObjectId(reminder_id)},
            {"$set": {"completed": True}}
        )

        # Broadcast over WebSocket
        reminder_data = {
            "id": reminder_id,
            "text": text,
            "date": when_local.isoformat(),
            "completed": True,
            "ai_response": ai_response
        }
        await broadcast_reminder(user_id, reminder_data)
        logger.info(f"Reminder {reminder_id} sent to user {user_id}")

    except Exception as e:
        logger.error(f"Error in schedule_reminder_for {reminder_id}: {e}", exc_info=True)
