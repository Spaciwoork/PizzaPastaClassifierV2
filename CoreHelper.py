import os
import yaml

from pymongo.uri_parser import parse_uri
from mongoengine import connect

dir_path = os.path.dirname(os.path.realpath(__file__))
config = yaml.safe_load(open(dir_path + "/config.yml"))

DEFAULT_DB_NAME = 'PizzaPastaGang'
MOCKED_DB_NAME = 'mongoenginetest'


class CoreHelper:
    mongo_connection = None
    mongo_db_name = None

    @staticmethod
    def mongo_uri():
        """Parse mongo uri."""
        return parse_uri(config['mongodb']['uri'])

    @staticmethod
    def mongodb_connect(uri=None):
        """
        Connect to mongodb (detect testing env).

        :param: uri: param you can overload to set a DB default configuration as 'mongomock://localhost'
        :rtype: MongoClient
        """
        uri_ = uri or config['mongodb']['uri']

        if 'mongomock' in uri_:
            CoreHelper.mongo_connection = connect(db=MOCKED_DB_NAME, host='mongomock://localhost', alias='default')
            return CoreHelper.mongo_connection

        testing = bool(os.getenv('TESTING', False))
        name = CoreHelper.mongo_uri()['database'] or DEFAULT_DB_NAME
        if testing:
            uri_ = f'{uri_}_test'
            name = f'{name}_test'

        CoreHelper.mongo_db_name = name

        CoreHelper.mongo_connection = connect(name, host=uri_, alias='default')
        return CoreHelper.mongo_connection
