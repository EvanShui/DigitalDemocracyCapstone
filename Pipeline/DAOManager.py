# Data Base Access Object Manager
#
# Responsible for maintaining connections to different Digital Democracy
#   databases for the purposes of scraping and downloading video and document
#   data.  Interactions with the database are logged using the Scraper Logger.
#
# The Database Manager is a singleton that can be easily retrieved.  Its
#   construction is lazy, which allows for the passing of a test parameter,
#   which indicates that the manager should mock the queries and statements.
#

import MySQLdb
import json
import os 
import traceback 
from scraper_logger import get_logger

logger = get_logger()

class DAOManager:
    dao_manager = None

    def __init__(self, test=False, log=True):
        """
        If |test| is True, the database connections are mocked.
        If |log| is true, all queries/statements are logged using the Scraper
            Logger.
        """
        self.test = test
        self.log = log

    def set_options(self, test=False, log=True):
        """
        If |test| is True, the database connections are mocked.
        If |log| is true, all queries/statements are logged using the Scraper
            Logger.
        """
        self.test = test
        self.log = log

    def get_dao(self):
        class DAO:
            def __init__(self, test=False, log=True):
                self.test = test
                self.log = log
                self.connections = []
          
            def __enter__(self):
                print self
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                self.cleanup()

            def cleanup(self):
                for conn in self.connections:
                    if conn:
                        print "closing.. ", conn
                        conn.close()

            def get_connection(self, sadb=False):
                print "test=", self.test
                print "SADB=", sadb
                if self.test:
                    return None
                # Connect to DDDB
                try:
                    logger.log("SADB=%s" % (sadb))
                    conn = self.__connect__(sadb)
                    logger.log("  connected to DB ")
                    self.connections.append(conn)
                    return conn
                except (IOError, MySQLdb.InterfaceError):
                    logger.error("Could not connect to DB", fatal=True)


            def __connect__(self, stateAgency):
                """
                Makes a connection with a database using the given credentials
                """

                return MySQLdb.connect(
                    host=get_env_var("SADB_HOST" if stateAgency else "DB_HOST"),
                    user=get_env_var("SADB_USER" if stateAgency else "DB_USER"),
                    passwd=get_env_var("SADB_PASSWORD" if stateAgency
                                       else "DB_PASSWORD"),
                    db=get_env_var("SADB_NAME" if stateAgency else "DB_NAME"),
                    charset='utf8',
                    use_unicode=True)

            def get_all_connections(self):
                ''' Returns all connections. '''
                print 'Returns all connections.=', self.connections
                return self.connections 

            def get_dddb_connection(self):
                ''' Returns connection to DDDB. '''
                return self.get_connection(False)

            def get_sadb_connection(self):
                ''' Returns connection to SADB. '''
                return self.get_connection(True)

        print "return DAO"
        return DAO(self.test, self.log) 

    def run_query(self, db_connection, query):
        ''' Runs a query '''
        if self.log is True:
            db = db_connection.get_host_info()
            logger.log("\n*** Executing query on %s:" % db)
            logger.log(query)

        if self.test is True:
            return []
        else:
            print db_connection
            cursor = db_connection.cursor()
        try:
            cursor.execute(query)
            return cursor.fetchall()
        except:
            traceback.print_exc(file=sys.stdout)
            logger.error("Statement failed:\n%s!" % query)
            return None 
        finally:
            cursor.close()

    def run_statement(self, db_connection, statement):
        '''
        Runs an insert or update statement.
        Returns whether the insert was successful.
        '''
        if self.log is True:
            db = db_connection.get_host_info()
            logger.log("\n*** Executing statement on %s:" % db)
            logger.log(statement)

        if self.test is True:
            return True
        else:
            cursor = db_connection.cursor()
            try:
                cursor.execute(statement)
                db_connection.commit()
                return True
            except MySQLdb.Error:
                traceback.print_exc(file=sys.stdout)
                db_connection.rollback()
                logger.error("Statement failed:\n%s!" % statement)
                return False
            finally:
                cursor.close()

    @staticmethod
    def create(test=False, log=True):
        if DAOManager.dao_manager is None:
            print "dao_manager does not exist!"
            DAOManager.dao_manager = DAOManager(test, log)
        else:
            print "dao_manager exists!"
        return DAOManager.dao_manager

#if __package__ is None:
if __package__ != 'app':
    import sys
    from os import path
    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
    from common.utility import get_env_var
else:
    from ..common.utility import get_env_var
