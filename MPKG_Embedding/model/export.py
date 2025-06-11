"""Functions to export graph database to json format and create config file for PBG training
"""

from .utils import connect_to_graphdb, logging
import os
import sys
import json

graph_connection = connect_to_graphdb()
GLOBAL_CONFIG = None
DATA_DIRECTORY = None
CHECKPOINT_DIRECTORY = None


def initialise_config(program):
    from .utils import load_config_program

    global GLOBAL_CONFIG
    global DATA_DIRECTORY
    global CHECKPOINT_DIRECTORY
    GLOBAL_CONFIG = load_config_program(program,"GLOBAL_CONFIG")
    cwd = os.getcwd()  # get current directory
    # default data
    DATA_DIRECTORY = os.path.join(
        cwd, GLOBAL_CONFIG["DATA_DIRECTORY"]
    )
    # default myproject/model
    CHECKPOINT_DIRECTORY = os.path.join(
        cwd, GLOBAL_CONFIG["PROJECT_NAME"]
    )


def create_folders():
    """creates folders for storing training data and model checkpoints as per the config.yml file
    """
    try:
        logging.info(f"""CREATING FOLDERS""")
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        os.makedirs(CHECKPOINT_DIRECTORY, exist_ok=True)
        logging.info(f"""Done.""")
    except Exception as e:
        logging.info("Could not create project directories")
        logging.info(e, exc_info=True)
        sys.exit(e)


def export_graph_to_json():
    """exports the graph database as a json file
    """
    try:
        export_file_name = GLOBAL_CONFIG["JSON_EXPORT_FILE"] + ".json"
        graph_file_path = os.path.abspath(
            os.path.join(
                DATA_DIRECTORY, export_file_name
            )  # default:  myproject/data/graph.json
        )
        logging.info(f"""EXPORTING GRAPH DATABASE TO {graph_file_path}...... """)
        graph_file_path_in_query = "file:///{}".format(graph_file_path.replace("\\","/"))
        query = (
            f"""CALL apoc.export.json.all('{graph_file_path_in_query}'"""
            + """,{batchSize:500})"""
        )
        graph_connection.run(query)
        if os.path.exists(graph_file_path):
            logging.info("Done...")
        else:
            logging.info("export failed! try again!")
    except Exception as e:
        logging.info(
            """error in exporting data. 
        Possible problemas may include incorrect url and credentials. 
        Or absence of apoc procedures. 
        Also make sure apoc settings are configured in neo4j.conf"""
        )
        logging.info(e, exc_info=True)
        sys.exit(e)



def export(program):
    """entry function for exporting graph data and creating PBG config.
    """
    try:
        initialise_config(program)
        logging.info(
            "-------------------------PREPARING FOR DATA EXPORT------------------------"
        )
        create_folders()  # create neccesary folders
        export_graph_to_json()  # export graph to json
        logging.info("Done....")
    except Exception as e:
        logging.info("error in export")
        logging.info(e, exc_info=True)
        sys.exit(e)
