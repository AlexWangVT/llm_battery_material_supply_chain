import yaml

class Config:
    _config = None  # class-level cache

    @classmethod
    def load(cls, path="config.yml"):
        if cls._config is None:
            with open(path, "r") as file:
                cls._config = yaml.safe_load(file)
        return cls._config

    @classmethod
    def get(cls, *keys):
        config = cls.load()
        for key in keys:
            config = config[key]
        return config