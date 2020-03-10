import pymongo

class ImageModel():
    def __init__(self):
        pass

    @staticmethod
    def create(uri):
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        db = myclient["PizzaPastaGang"]
        collection = db["Image"]

        toInsert = {"uri": uri}

        x = collection.insert_one(toInsert)