"""
General utility functions
"""
from py2neo import Graph
import yaml
from yaml.loader import Loader
import logging
import os
import time

CONFIG_FILE_PATH = "../MPKG_Embedding/config.yml"
logging.basicConfig(format="%(asctime)s - %(message)s", level=20)


def load_config(subconfig_name: str = None):
    """Loads config in YAML file. Can load one part of the config or
    the entire config if subconfig_name is not given
    subconfig_name can be from: [GLOBAL_CONFIG, GRAPH_DATABASE, OPTIONAL_PBG_SETTINGS]

    Keyword Arguments:
        subconfig_name {str} -- part of the config to load (default: {None})

    Returns:
        [dict] -- [config file]
    """
    try:
        config_file = open(CONFIG_FILE_PATH).read()
        config = yaml.load(config_file, Loader=Loader)
        if subconfig_name is not None:
            subconfig = config[subconfig_name]
            return subconfig
        return config
    except Exception as e:
        logging.info(f"Error in loading config : {e}", exc_info=True)

def load_config_program(program,subconfig_name: str = None):
    try:
        PROGRAM_CONFIG_PATH = "{}/config.yml".format(program)
        config_file = open(PROGRAM_CONFIG_PATH).read()
        config = yaml.load(config_file, Loader=Loader)
        if subconfig_name is not None:
            subconfig = config[subconfig_name]
            return subconfig
        return config
    except Exception as e:
        logging.info(f"Error in loading config : {e}", exc_info=True)

def connect_to_graphdb():
    """connect to graph database
    Returns:
        [type] -- [connection to graph database]
    """
    try:
        graph_config = load_config("GRAPH_DATABASE")
        url = graph_config["URL"]
        # url = f"bolt://localhost:{tunnel.local_bind_port}"
        username = graph_config["USERNAME"]
        password = graph_config["PASSWORD"]
        graph_db = Graph(url, user=username, password=str(password))
        # graph_db = Graph(url, auth = (username, password))
        return graph_db
    except Exception as e:
        logging.info(f"Error in connecting to graph database : {e}", exc_info=True)


def test_db_connection():
    try:
        logging.info("CONNECTING TO DATABASE.........")
        graph_connection = connect_to_graphdb()
        graph_connection.run("""MATCH (n:series) WHERE n.name = "G273" RETURN n""")
        return True
    except Exception as e:
        logging.info(
            f"Could not connect to Database. Please check your credentials:\n{e}"
        )
        return False


def update_config(**kwargs):
    try:
        logging.info("-----------------UPDATING CONFIG-----------------")
        default_config = load_config()
        for key, value in kwargs.items():
            if default_config["GLOBAL_CONFIG"].get(key.upper()):
                default_config["GLOBAL_CONFIG"][key.upper()] = value
            elif default_config["GRAPH_DATABASE"].get(key.upper()):
                default_config["GRAPH_DATABASE"][key.upper()] = value
            elif not default_config["EMBEDDING_SETTINGS"].get(key) == None:
                default_config["EMBEDDING_SETTINGS"][key] = value
            elif default_config.get(key):
                default_config[key] = value
        MODEL_CONFIG_PATH = os.path.join(
            os.getcwd(),
            default_config["GLOBAL_CONFIG"]["PROJECT_NAME"],
            default_config["GLOBAL_CONFIG"]["CONFIG_NAME"],
        )
        CHECKPOINT_DIRECTORY = os.path.join(
            os.getcwd(),
            default_config["GLOBAL_CONFIG"]["PROJECT_NAME"],
        )
        os.makedirs(CHECKPOINT_DIRECTORY, exist_ok=True)
        with open(MODEL_CONFIG_PATH, "w") as f:
            yaml.dump(default_config, f)
        # with open(CONFIG_FILE_PATH, "w") as f:
        #     yaml.dump(default_config, f)
        logging.info("Done....")
    except Exception as e:
        logging.info(f"Error in updating config : {e}", exc_info=True)


def get_checkpoint_version():
    """returns the latest version of the embeddings

    Returns:
        [int] -- version of the embeddings
    """
    try:
        GLOBAL_CONFIG = load_config("GLOBAL_CONFIG")
        checkpoint_version_file = os.path.join(
            GLOBAL_CONFIG["PROJECT_NAME"],
            GLOBAL_CONFIG["CHECKPOINT_DIRECTORY"],
            "checkpoint_version.txt",
        )
        with open(checkpoint_version_file, "r") as f:
            version = f.read()
        f.close()
        version = int(version.split()[0].strip())
        logging.info(f"Latest checkpoint version: {version}")
        return version
    except Exception as e:
        logging.error(f"Could locate checkpoint version file: {e}", exc_info=True)
