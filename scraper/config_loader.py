import configparser


class ConfigLoader:
    @staticmethod
    def load_config(filename):
        config = configparser.ConfigParser()
        with open(filename) as f:
            config.read_file(f)
        return config
