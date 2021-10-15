class DatasetConfig:
  config = None

  @classmethod
  def maybe_set_config(cls, config, override=False):
    if config is not None:
      if cls.config is None or override:
        print(f'[maybe_set_config] {"Overriding" if override else "Setting"} PbN global config.')
        cls.config = config
      else:
        print(f'[maybe_set_config] PbN global config is already set. Ignoring second attempt.')
