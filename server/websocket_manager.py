from fastapi import WebSocket
import logging

logger = logging.getLogger(__name__)

active_connections: list[tuple[str, WebSocket]] = []

async def broadcast_reminder(user_id: str, reminder_data: dict):
    for conn_user_id, connection in active_connections[:]:
        if conn_user_id == user_id:
            try:
                await connection.send_json(reminder_data)
            except Exception as e:
                logger.error(f"Error sending to WebSocket for user {user_id}: {e}")
                active_connections.remove((conn_user_id, connection))