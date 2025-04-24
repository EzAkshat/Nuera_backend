from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_URI)
db = client["Nuera"]

# Collections
users_collection = db["users"]
chats_collection = db["chats"]
reminders_collection = db["reminders"]

# No startup event needed since unique index creation is removed
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_URI)
db = client["Nuera"]

# Collections
users_collection = db["users"]
chats_collection = db["chats"]
reminders_collection = db["reminders"]

# No need for unique index creation
async def startup_event():
    print("Database connection established.")