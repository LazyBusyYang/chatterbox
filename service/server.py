import asyncio
import io
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from urllib.parse import quote

import torch
import torchaudio as ta
import uvicorn
from fastapi import (
    APIRouter,
    FastAPI,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.mtl_tts import SUPPORTED_LANGUAGES
from chatterbox.models.s3gen import S3GEN_SR

from .utils import setup_logger


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_TEXT_LENGTH = 500
MIN_PROMPT_SECONDS = 2
MAX_PROMPT_SECONDS = 30
UPLOAD_CHUNK_BYTES = 1024 * 1024


class ListVoiceNameResponse(BaseModel):
    """Response model for listing available voice names.

    Contains a dictionary mapping voice keys to their display names.
    """
    voice_names: dict[str, str]


class GenerateAudioRequest(BaseModel):
    """Request model for generating audio from text.

    Contains the text to synthesize and the voice key to use for synthesis.
    """
    text: str
    voice_key: str


class FastAPIServer:
    """FastAPI server for text-to-speech audio generation.

    Provides HTTP endpoints for listing available voices and generating
    audio from text using the ChatterboxMultilingualTTS model.
    """

    def __init__(
        self,
        audio_prompts_dir: str,
        checkpoint_dir: str | None = None,
        device: str | None = None,
        enable_cors: bool = False,
        host: str = '0.0.0.0',
        port: int = 80,
        startup_event_listener: None | list = None,
        shutdown_event_listener: None | list = None,
        logger_cfg: None | dict = None,
    ) -> None:
        """Initialize the FastAPI server.

        Sets up the FastAPI application, configures CORS if enabled,
        registers event listeners, and initializes the TTS model.

        Args:
            audio_prompts_dir (str):
                Directory path containing audio prompt files.
                Files should be named as '{voice_key}_{language_id}.wav'.
            checkpoint_dir (str | None, optional):
                Directory path to load TTS model checkpoint from.
                If None, loads pretrained model from HuggingFace.
                Defaults to None.
            device (str | None, optional):
                Device to run the model on ('cuda', 'mps', or 'cpu').
                If None, automatically selects based on availability.
                Defaults to None.
            enable_cors (bool, optional):
                Whether to enable CORS middleware. Defaults to False.
            host (str, optional):
                Host address to bind the server to. Defaults to '0.0.0.0'.
            port (int, optional):
                Port number to bind the server to. Defaults to 80.
            startup_event_listener (None | list, optional):
                List of startup event listener functions.
                Defaults to None.
            shutdown_event_listener (None | list, optional):
                List of shutdown event listener functions.
                Defaults to None.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed
                description. Logger name will use the class name.
                Defaults to None.
        """
        logger_name = self.__class__.__name__
        if logger_cfg is None:
            logger_cfg = dict(logger_name=logger_name)
        else:
            logger_cfg = logger_cfg.copy()
            logger_cfg["logger_name"] = logger_name
        self.logger_cfg = logger_cfg
        self.logger = setup_logger(**logger_cfg)
        self.audio_prompts_dir = audio_prompts_dir
        self._load_audio_prompts()
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        # for fastapi
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.enable_cors = enable_cors
        if self.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=['*'],
                allow_credentials=True,
                allow_methods=['*'],
                allow_headers=['*'],
            )
        if startup_event_listener is not None:
            for listener in startup_event_listener:
                self.app.add_event_handler('startup', listener)
        if shutdown_event_listener is not None:
            for listener in shutdown_event_listener:
                self.app.add_event_handler('shutdown', listener)
        self._build_tts_model()
        self.last_audio_prompt_key: str | None = None
        self.model_lock = Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.audio_thread_pool = ThreadPoolExecutor(max_workers=2)

    def _load_audio_prompts(self) -> None:
        """Load audio prompt files from the configured directory.

        Scans the audio prompts directory for WAV files following the naming
        convention '{voice_key}_{language_id}.wav'. Valid files are registered
        in the audio_prompts dictionary for use in voice synthesis.
        """
        self.audio_prompts = dict()
        for file in os.listdir(self.audio_prompts_dir):
            if file.endswith('.wav'):
                file_name = file.split('.')[0]
                splits = file_name.split('_')
                if len(splits) != 2:
                    self.logger.warning(f"Invalid file name: {file_name}, skipping")
                    continue
                voice_key, language_id = splits
                self.audio_prompts[file_name] = dict(
                    name=voice_key,
                    language_id=language_id,
                    path=os.path.join(self.audio_prompts_dir, file)
                )

    def _build_tts_model(self) -> None:
        """Initialize and load the TTS model.

        Determines the appropriate device (CUDA, MPS, or CPU) based on
        availability if device is not specified. Loads the TTS model from
        local checkpoint directory if provided, otherwise loads the
        pretrained model from HuggingFace.
        """
        if self.device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            self.logger.info(f"Device not specified, using device automatically: {device}")
        else:
            device = self.device
        if self.checkpoint_dir is not None and os.path.exists(self.checkpoint_dir):
            self.tts_model = ChatterboxMultilingualTTS.from_local(self.checkpoint_dir, device)
        else:
            msg = "Checkpoint directory not specified or does not exist, using pretrained model"
            self.logger.info(msg)
            self.tts_model = ChatterboxMultilingualTTS.from_pretrained(device)

    def _encode_tensor_to_wav(self, tensor_wav: torch.Tensor) -> io.BytesIO:
        """Encode a generated waveform tensor as PCM WAV bytes.

        Args:
            tensor_wav (torch.Tensor):
                Generated waveform with channels first.

        Returns:
            io.BytesIO:
                BytesIO buffer containing a PCM 16-bit WAV file.
        """
        wav_io = io.BytesIO()
        ta.save(
            uri=wav_io,
            src=tensor_wav,
            sample_rate=self.tts_model.sr,
            channels_first=True,
            format='wav',
            encoding='PCM_S',
            bits_per_sample=16
        )
        return wav_io

    def _generate_audio(self, voice_key: str, text: str) -> io.BytesIO:
        """Generate audio from text using the specified voice.

        This method handles the core audio generation logic, including voice
        prompt preparation, text-to-speech synthesis, and audio format conversion.
        It uses thread-safe locking to ensure model access is serialized.

        Args:
            voice_key (str):
                Voice key identifier matching an audio prompt file name
                (without extension).
            text (str):
                Text content to synthesize into speech.

        Returns:
            io.BytesIO:
                BytesIO buffer containing the generated WAV audio file
                in PCM format (16-bit signed integer).
        """
        with self.model_lock:
            if self.last_audio_prompt_key is None or \
                    self.last_audio_prompt_key != voice_key:
                audio_prompt_path = self.audio_prompts[voice_key]['path']
                prepare_start_time = time.time()
                self.tts_model.prepare_conditionals(audio_prompt_path)
                prepare_end_time = time.time()
                self.logger.info(
                    f"Prepare time: {prepare_end_time - prepare_start_time:.2f} "
                    f"seconds for {voice_key}"
                )
                self.last_audio_prompt_key = voice_key
            language_id = self.audio_prompts[voice_key]['language_id']
            generate_start_time = time.time()
            tensor_wav = self.tts_model.generate(text, language_id=language_id)
            generate_end_time = time.time()
            # Calculate duration directly from tensor shape
            duration = tensor_wav.shape[-1] / self.tts_model.sr
            wav_io = self._encode_tensor_to_wav(tensor_wav)
        self.logger.info(
            f"Generate time: {generate_end_time - generate_start_time:.2f} seconds " +
            f"for {voice_key}, duration: {duration:.2f} seconds")
        return wav_io

    def _convert_audio_prompt(self, input_path: Path, output_path: Path) -> float:
        """Convert an uploaded prompt audio file into the model reference format.

        Args:
            input_path (Path):
                Path to the uploaded temporary file.
            output_path (Path):
                Path where the converted WAV file should be written.

        Returns:
            float:
                Converted WAV duration in seconds.

        Raises:
            HTTPException:
                Raised with status 400 when ffmpeg cannot decode/convert the file
                or the converted audio does not meet validation requirements.
        """
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(S3GEN_SR),
            "-f",
            "wav",
            str(output_path),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            self.logger.exception("ffmpeg is not installed")
            raise HTTPException(
                status_code=500,
                detail="Audio conversion backend is unavailable",
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or "Uploaded audio could not be decoded"
            self.logger.warning(f"ffmpeg failed to convert uploaded audio: {detail}")
            raise HTTPException(
                status_code=400,
                detail="Uploaded audio could not be decoded",
            ) from exc

        try:
            audio_info = ta.info(str(output_path))
            duration = audio_info.num_frames / audio_info.sample_rate
        except Exception as exc:
            self.logger.warning(f"Failed to inspect converted audio prompt: {exc}")
            raise HTTPException(
                status_code=400,
                detail="Converted audio could not be inspected",
            ) from exc

        if audio_info.sample_rate <= 0 or audio_info.num_frames <= 0:
            raise HTTPException(status_code=400, detail="Uploaded audio is empty")
        if audio_info.sample_rate != S3GEN_SR or audio_info.num_channels != 1:
            raise HTTPException(status_code=400, detail="Converted audio has an invalid format")
        if duration < MIN_PROMPT_SECONDS or duration > MAX_PROMPT_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Audio prompt duration must be between "
                    f"{MIN_PROMPT_SECONDS} and {MAX_PROMPT_SECONDS} seconds"
                ),
            )
        return duration

    def _generate_audio_with_prompt(
        self,
        audio_prompt_path: str,
        language_id: str,
        text: str,
    ) -> io.BytesIO:
        """Generate audio using a temporary uploaded reference prompt.

        Args:
            audio_prompt_path (str):
                Path to the converted temporary WAV prompt.
            language_id (str):
                Supported Chatterbox language identifier.
            text (str):
                Text content to synthesize.

        Returns:
            io.BytesIO:
                BytesIO buffer containing generated PCM WAV audio.
        """
        with self.model_lock:
            self.last_audio_prompt_key = None
            generate_start_time = time.time()
            tensor_wav = self.tts_model.generate(
                text,
                language_id=language_id,
                audio_prompt_path=audio_prompt_path,
            )
            generate_end_time = time.time()
            duration = tensor_wav.shape[-1] / self.tts_model.sr
            wav_io = self._encode_tensor_to_wav(tensor_wav)
        self.logger.info(
            f"Generate time: {generate_end_time - generate_start_time:.2f} seconds " +
            f"for uploaded prompt, duration: {duration:.2f} seconds")
        return wav_io

    async def _save_uploaded_audio_prompt(self, upload: UploadFile, input_path: Path) -> int:
        """Save an uploaded prompt file with a strict size limit.

        Args:
            upload (UploadFile):
                Uploaded prompt audio file.
            input_path (Path):
                Temporary destination path.

        Returns:
            int:
                Number of bytes written.

        Raises:
            HTTPException:
                Raised with status 400 when the upload is empty or too large.
        """
        loop = asyncio.get_event_loop()
        written = 0
        with input_path.open("wb") as file:
            while True:
                chunk = await upload.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=400,
                        detail="Audio prompt file must be 20MB or smaller",
                    )
                await loop.run_in_executor(self.audio_thread_pool, file.write, chunk)
        if written == 0:
            raise HTTPException(status_code=400, detail="Audio prompt file must not be empty")
        return written

    def _validate_text(self, text: str | None) -> str:
        """Validate and normalize text input for synthesis.

        Args:
            text (str):
                Raw user-provided text.

        Returns:
            str:
                Trimmed text.

        Raises:
            HTTPException:
                Raised with status 400 when text is empty or too long.
        """
        if text is None:
            raise HTTPException(status_code=400, detail="Text must not be empty")
        normalized = text.strip()
        if not normalized:
            raise HTTPException(status_code=400, detail="Text must not be empty")
        if len(normalized) > MAX_TEXT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Text must be {MAX_TEXT_LENGTH} characters or fewer",
            )
        return normalized

    def _validate_language_id(self, language_id: str | None) -> str:
        """Validate and normalize a Chatterbox language identifier.

        Args:
            language_id (str):
                Raw language identifier.

        Returns:
            str:
                Lowercase language identifier.

        Raises:
            HTTPException:
                Raised with status 400 when the language is unsupported.
        """
        if language_id is None:
            raise HTTPException(status_code=400, detail="language_id is required")
        normalized = language_id.strip().lower()
        if normalized not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(sorted(SUPPORTED_LANGUAGES))
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language_id. Supported languages: {supported_langs}",
            )
        return normalized

    def _add_api_routes(self, router: APIRouter) -> None:
        """Add API routes to the router.

        This method registers all HTTP endpoints with the provided FastAPI router,
        including voice listing, audio generation, health checks, and root redirect.

        Args:
            router (APIRouter):
                FastAPI router to add routes to.
        """
        # GET routes
        router.add_api_route(
            "/api/v1/list_voice_names",
            self.list_voice_names,
            methods=["GET"],
            response_model=ListVoiceNameResponse,
        )
        router.add_api_route(
            "/api/v1/generate_audio",
            self.generate_audio,
            methods=["POST"]
        )
        router.add_api_route(
            "/api/v1/generate_audio_with_prompt",
            self.generate_audio_with_prompt,
            methods=["POST"]
        )
        router.add_api_route(
            "/",
            self.root,
            methods=["GET"],
        )
        router.add_api_route(
            '/health',
            endpoint=self.health,
            status_code=200,
            methods=['GET'],
        )

    def run(self) -> None:
        """Run this FastAPI service according to configuration.

        Registers all API routes and starts the uvicorn server with the
        configured host, port, and application settings. This method
        blocks until the server is stopped.
        """
        router = APIRouter()
        self._add_api_routes(router)
        self.app.include_router(router)
        uvicorn.run(self.app, host=self.host, port=self.port)

    def root(self) -> RedirectResponse:
        """Redirect to API documentation.

        Returns:
            RedirectResponse:
                Redirect response to /docs endpoint.
        """
        return RedirectResponse(url="/docs")

    async def health(self) -> JSONResponse:
        """Health check endpoint for service monitoring.

        This endpoint provides a simple health check that returns
        an 'OK' status to indicate the service is running properly.
        Used by load balancers and monitoring systems.

        Returns:
            JSONResponse:
                JSON response containing 'OK' status string.
        """
        resp = JSONResponse(content='OK')
        return resp

    async def list_voice_names(self) -> ListVoiceNameResponse:
        """List all available voice names.

        Scans the audio prompts directory and returns all available
        voice configurations that can be used for text-to-speech synthesis.

        Returns:
            ListVoiceNameResponse:
                Response containing a dictionary mapping voice keys
                to their display names.
        """
        voice_names = dict()
        for key in self.audio_prompts:
            voice_names[key] = self.audio_prompts[key]['name']
        resp = ListVoiceNameResponse(voice_names=voice_names)
        return resp

    async def generate_audio(self, request: GenerateAudioRequest) -> Response:
        """Generate audio from text using the specified voice.

        Synthesizes speech from the input text using the TTS model with
        the specified voice. If the voice prompt needs to be loaded,
        it will be prepared before generation. The generated audio is
        returned as a WAV file with appropriate download headers.

        Args:
            request (GenerateAudioRequest):
                Request containing the text to synthesize and the voice
                key to use.

        Returns:
            Response:
                HTTP response containing the generated audio as a WAV
                file with appropriate headers for download.
        """
        if request.voice_key not in self.audio_prompts:
            msg = f"Voice key {request.voice_key} not found"
            self.logger.error(msg)
            raise HTTPException(status_code=404, detail=msg)
        loop = asyncio.get_event_loop()
        wav_io = await loop.run_in_executor(
            self.thread_pool, self._generate_audio, request.voice_key, request.text)
        wav_io.seek(0)
        resp = Response(
            content=wav_io.getvalue(),
            media_type="audio/wav")
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        filename = f'{request.voice_key}_{timestamp_str}.wav'
        # Use RFC 5987 standard encoding for filename to support non-ASCII characters
        encoded_filename = quote(filename, safe='')
        resp.headers['Content-Disposition'] = f"attachment; filename*=UTF-8''{encoded_filename}"
        return resp

    async def generate_audio_with_prompt(
        self,
        text: str | None = Form(None),
        language_id: str | None = Form(None),
        audio_prompt: UploadFile | None = File(None),
    ) -> Response:
        """Generate audio using an uploaded temporary reference prompt.

        Args:
            text (str):
                Text to synthesize. Must be non-empty and at most 500 chars.
            language_id (str):
                Supported language identifier for multilingual synthesis.
            audio_prompt (UploadFile):
                Uploaded reference audio file. It is decoded by ffmpeg and
                converted to mono WAV at S3GEN_SR before model inference.

        Returns:
            Response:
                HTTP response containing generated audio as a WAV file.
        """
        normalized_text = self._validate_text(text)
        normalized_language_id = self._validate_language_id(language_id)
        if audio_prompt is None:
            raise HTTPException(status_code=400, detail="audio_prompt file is required")

        with tempfile.TemporaryDirectory(prefix="chatterbox_prompt_") as temp_dir:
            temp_path = Path(temp_dir)
            uploaded_path = temp_path / "uploaded_audio"
            converted_path = temp_path / "prompt.wav"
            await self._save_uploaded_audio_prompt(audio_prompt, uploaded_path)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.audio_thread_pool,
                self._convert_audio_prompt,
                uploaded_path,
                converted_path,
            )
            try:
                wav_io = await loop.run_in_executor(
                    self.thread_pool,
                    self._generate_audio_with_prompt,
                    str(converted_path),
                    normalized_language_id,
                    normalized_text,
                )
            except HTTPException:
                raise
            except Exception as exc:
                self.logger.exception("Failed to generate audio with uploaded prompt")
                raise HTTPException(status_code=500, detail="Failed to generate audio") from exc

        wav_io.seek(0)
        resp = Response(
            content=wav_io.getvalue(),
            media_type="audio/wav")
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        filename = f'uploaded_prompt_{normalized_language_id}_{timestamp_str}.wav'
        encoded_filename = quote(filename, safe='')
        resp.headers['Content-Disposition'] = f"attachment; filename*=UTF-8''{encoded_filename}"
        return resp
