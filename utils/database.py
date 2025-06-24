from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDB:
    _instance = None
    _client = None
    _db = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDB, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            self.connect()

    def connect(self):
        """Establish connection to MongoDB"""
        try:
            # Load environment variables
            load_dotenv()
            
            # Get MongoDB URI from environment variable or use default
            mongo_uri = os.getenv('MONGODB_URI', 'mongodb+srv://pkbhopi132:t6FAQ3U55iOMFkPu@truedoc.5uftss9.mongodb.net/')
            
            # Create MongoDB client
            self._client = MongoClient(
                mongo_uri,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            
            # Test the connection
            self._client.admin.command('ping')
            logger.info("Successfully connected to MongoDB!")
            
            # Get database instance
            self._db = self._client.get_database('Docs')
            
        except ConnectionFailure as e:  
            logger.error(f"Could not connect to MongoDB: {e}")
            raise
        except ServerSelectionTimeoutError as e:
            logger.error(f"Server selection timeout: {e}")
            raise
        except Exception as e:
            logger.error(f"An error occurred while connecting to MongoDB: {e}")
            raise

    def get_database(self):
        """Get database instance"""
        if self._db is None:
            self.connect()
        return self._db

    def get_collection(self, collection_name):
        """Get collection instance"""
        if self._db is None:
            self.connect()
        return self._db[collection_name]

    def close(self):
        """Close MongoDB connection"""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            logger.info("MongoDB connection closed")

    def __del__(self):
        """Destructor to ensure connection is closed"""
        self.close()

# Example usage:
if __name__ == "__main__":
    try:
        # Get MongoDB instance
        mongo = MongoDB()
        
        # Get database
        db = mongo.get_database()
        
        # Get collection
        collection = mongo.get_collection('documents')
        
        # Example: Insert a document
        result = collection.insert_one({
            "test": "Hello MongoDB!"
        })
        print(f"Inserted document ID: {result.inserted_id}")
        
        # Example: Find documents
        documents = collection.find()
        for doc in documents:
            print(doc)
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        # Close connection
        mongo.close() 