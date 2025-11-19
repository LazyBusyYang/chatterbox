# CRITICAL = 50
# ERROR = 40
# WARNING = 30
# INFO = 20
# DEBUG = 10
# NOTSET = 0
__logger_cfg__ = dict(
    logger_name="root",
    file_level=10,
    console_level=20,
    logger_path='logs/server.log',
)

type = 'FastAPIServer'
audio_prompts = dict(
    keqing_zh=dict(
        name="刻晴-中文",
        language_id="zh",
        path="data/keqing_zh_12min.wav"
    ),
    furina_zh=dict(
        name="芙宁娜-中文",
        language_id="zh",
        path="data/furina_zh_16min.wav"
    ),
    hutao_zh=dict(
        name="胡桃-中文",
        language_id="zh",
        path="data/hutao_zh_6min.wav"
    ),
)
enable_cors = True
host = '0.0.0.0'
port = 18085
logger_cfg = __logger_cfg__
