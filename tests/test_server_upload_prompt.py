import asyncio
import io
import subprocess
import sys
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from unittest.mock import patch

class HTTPException(Exception):
    def __init__(self, status_code, detail=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class APIRouter:
    def add_api_route(self, *_args, **_kwargs):
        pass


class FastAPI:
    def add_middleware(self, *_args, **_kwargs):
        pass

    def add_event_handler(self, *_args, **_kwargs):
        pass

    def include_router(self, *_args, **_kwargs):
        pass


class Response:
    def __init__(self, content=None, media_type=None):
        self.content = content
        self.media_type = media_type
        self.headers = {}


class BaseModel:
    pass


def form_or_file(default=None):
    return default


sys.modules.setdefault(
    "fastapi",
    types.SimpleNamespace(
        APIRouter=APIRouter,
        FastAPI=FastAPI,
        File=form_or_file,
        Form=form_or_file,
        HTTPException=HTTPException,
        Response=Response,
        UploadFile=object,
    ),
)
sys.modules.setdefault("fastapi.middleware", types.ModuleType("fastapi.middleware"))
sys.modules.setdefault(
    "fastapi.middleware.cors",
    types.SimpleNamespace(CORSMiddleware=object),
)
sys.modules.setdefault(
    "fastapi.responses",
    types.SimpleNamespace(JSONResponse=Response, RedirectResponse=Response),
)
sys.modules.setdefault("pydantic", types.SimpleNamespace(BaseModel=BaseModel))
sys.modules.setdefault(
    "torch",
    types.SimpleNamespace(
        Tensor=object,
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        ),
    ),
)
sys.modules.setdefault(
    "torchaudio",
    types.SimpleNamespace(save=lambda **_kwargs: None, info=lambda _path: None),
)
sys.modules.setdefault("uvicorn", types.SimpleNamespace(run=lambda *_args, **_kwargs: None))

chatterbox_module = types.ModuleType("chatterbox")
mtl_tts_module = types.ModuleType("chatterbox.mtl_tts")
mtl_tts_module.ChatterboxMultilingualTTS = object
mtl_tts_module.SUPPORTED_LANGUAGES = {"en": "English", "zh": "Chinese"}
models_module = types.ModuleType("chatterbox.models")
s3gen_module = types.ModuleType("chatterbox.models.s3gen")
s3gen_module.S3GEN_SR = 24000
sys.modules.setdefault("chatterbox", chatterbox_module)
sys.modules.setdefault("chatterbox.mtl_tts", mtl_tts_module)
sys.modules.setdefault("chatterbox.models", models_module)
sys.modules.setdefault("chatterbox.models.s3gen", s3gen_module)

from service import server as server_mod
from service.server import FastAPIServer


class DummyLogger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass

    def exception(self, *_args, **_kwargs):
        pass


class DummyUpload:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    async def read(self, _size):
        if not self.chunks:
            return b""
        return self.chunks.pop(0)


class DummyTensor:
    shape = (1, 24000)


class DummyTTSModel:
    sr = 24000

    def __init__(self):
        self.calls = []

    def generate(self, text, language_id, audio_prompt_path=None):
        self.calls.append(
            {
                "text": text,
                "language_id": language_id,
                "audio_prompt_path": audio_prompt_path,
            }
        )
        return DummyTensor()


def make_server():
    server = object.__new__(FastAPIServer)
    server.logger = DummyLogger()
    server.model_lock = Lock()
    server.audio_thread_pool = ThreadPoolExecutor(max_workers=1)
    server.tts_model = DummyTTSModel()
    server.last_audio_prompt_key = "cached_voice"
    server._encode_tensor_to_wav = lambda _tensor: io.BytesIO(b"wav")
    return server


class UploadPromptValidationTest(unittest.TestCase):
    def test_text_validation_rejects_empty_and_long_text(self):
        server = make_server()

        with self.assertRaises(HTTPException) as empty_exc:
            server._validate_text("   ")
        self.assertEqual(empty_exc.exception.status_code, 400)

        with self.assertRaises(HTTPException) as long_exc:
            server._validate_text("x" * (server_mod.MAX_TEXT_LENGTH + 1))
        self.assertEqual(long_exc.exception.status_code, 400)

    def test_language_validation_rejects_unsupported_language(self):
        server = make_server()

        self.assertEqual(server._validate_language_id(" EN "), "en")
        with self.assertRaises(HTTPException) as exc:
            server._validate_language_id("unknown")
        self.assertEqual(exc.exception.status_code, 400)

    def test_save_uploaded_audio_prompt_rejects_empty_and_oversize_files(self):
        server = make_server()

        async def run_case():
            with self.assertRaises(HTTPException) as empty_exc:
                await server._save_uploaded_audio_prompt(DummyUpload([]), Path("/tmp/unused"))
            self.assertEqual(empty_exc.exception.status_code, 400)

            with patch.object(server_mod, "MAX_UPLOAD_BYTES", 3):
                with self.assertRaises(HTTPException) as oversize_exc:
                    await server._save_uploaded_audio_prompt(
                        DummyUpload([b"ab", b"cd"]),
                        Path("/tmp/unused"),
                    )
                self.assertEqual(oversize_exc.exception.status_code, 400)

        asyncio.run(run_case())

    def test_convert_audio_prompt_rejects_ffmpeg_decode_failure(self):
        server = make_server()
        error = subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffmpeg"],
            stderr="invalid data",
        )

        with patch.object(server_mod.subprocess, "run", side_effect=error):
            with self.assertRaises(HTTPException) as exc:
                server._convert_audio_prompt(Path("input.bin"), Path("output.wav"))
        self.assertEqual(exc.exception.status_code, 400)

    def test_convert_audio_prompt_rejects_invalid_duration_and_format(self):
        server = make_server()
        valid_run = subprocess.CompletedProcess(args=["ffmpeg"], returncode=0)

        with patch.object(server_mod.subprocess, "run", return_value=valid_run):
            with patch.object(
                server_mod.ta,
                "info",
                return_value=types.SimpleNamespace(
                    num_frames=server_mod.S3GEN_SR,
                    sample_rate=server_mod.S3GEN_SR,
                    num_channels=1,
                ),
            ):
                with self.assertRaises(HTTPException) as short_exc:
                    server._convert_audio_prompt(Path("input.bin"), Path("output.wav"))
                self.assertEqual(short_exc.exception.status_code, 400)

            with patch.object(
                server_mod.ta,
                "info",
                return_value=types.SimpleNamespace(
                    num_frames=server_mod.S3GEN_SR * 5,
                    sample_rate=server_mod.S3GEN_SR,
                    num_channels=2,
                ),
            ):
                with self.assertRaises(HTTPException) as format_exc:
                    server._convert_audio_prompt(Path("input.bin"), Path("output.wav"))
                self.assertEqual(format_exc.exception.status_code, 400)

    def test_generate_audio_with_prompt_does_not_update_cached_voice_key(self):
        server = make_server()

        wav = server._generate_audio_with_prompt(
            audio_prompt_path="/tmp/prompt.wav",
            language_id="en",
            text="hello",
        )

        self.assertEqual(wav.getvalue(), b"wav")
        self.assertEqual(server.last_audio_prompt_key, "cached_voice")
        self.assertEqual(
            server.tts_model.calls,
            [
                {
                    "text": "hello",
                    "language_id": "en",
                    "audio_prompt_path": "/tmp/prompt.wav",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
