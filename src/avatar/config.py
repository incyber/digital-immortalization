"""Process configuration.

One Settings object, populated from the environment and an optional .env file.
No other module in this package reads os.environ directly - that rule is what
keeps the difference between a laptop and a GPU server to a set of values
rather than a set of branches.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Transport. The docker-compose dev server issues this key/secret pair.
    livekit_url: str = "ws://localhost:7880"
    livekit_api_key: str = "devkey"
    livekit_api_secret: str = "secret"

    # Language model, reached through an OpenAI-compatible endpoint. Ollama
    # serves one at /v1, so the same client works against a local model or a
    # hosted provider with no code change.
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "llama3.2:3b"
    llm_api_key: str = "ollama"  # ignored by Ollama; required by the client
    llm_provider_name: str = "ollama"

    # Fallback language model, used when the primary runs out of quota
    # mid-conversation. Any OpenAI-compatible endpoint works; the common ones:
    #   Groq    https://api.groq.com/openai/v1
    #   xAI     https://api.x.ai/v1
    #   Gemini  https://generativelanguage.googleapis.com/v1beta/openai
    # Left empty means no fallback rather than a broken one.
    fallback_llm_provider_name: str = "groq"
    fallback_llm_base_url: str = "https://api.groq.com/openai/v1"
    fallback_llm_api_key: str = ""
    # Measured against Groq's catalogue on 2026-08-24: llama-3.3-70b-versatile
    # has been retired, the qwen3 model leaks <think> blocks into content, and
    # groq/compound routes onto gpt-oss-120b so it shares its rate limit
    # without adding anything. gpt-oss-120b answers Spanish in persona in
    # ~0.6s provided reasoning_effort is turned down.
    fallback_llm_model: str = "openai/gpt-oss-120b"

    # Vision model. Ollama's native endpoint, not the OpenAI shim, because the
    # shim's image handling varies by version.
    vlm_base_url: str = "http://localhost:11434"
    vlm_model: str = "qwen2.5vl:3b"

    # Backend selection. See the execution matrix in the design document.
    #   stt_backend:      "mlx" (Metal, Apple Silicon) | "faster" (CTranslate2)
    #   renderer_backend: "viseme" (CPU) | "musetalk" (CUDA, sub-project 2)
    stt_backend: str = "mlx"
    # Measured on an M3 Max, end of speech to first avatar audio:
    #   whisper-base-mlx           fastest, but mis-hears short Spanish
    #                              ("Hola" -> "Bola") and derails the reply
    #   whisper-small-mlx          1.9s total, 1.4s in STT, accurate
    #   distil-whisper-large-v3    2.5s total, 1.7s in STT
    #   whisper-large-v3-turbo     2.9s total, 2.1s in STT
    # Small is the local default: it is the smallest model that transcribes
    # short Spanish utterances correctly. Cloud uses large-v3 on a GPU, where
    # the accuracy is free.
    stt_model: str = "mlx-community/whisper-small-mlx"
    # Language hint for STT. Empty means autodetect, which is unreliable on
    # single short utterances.
    stt_language: str = "es"
    renderer_backend: str = "viseme"

    # Piper voice. Downloaded once into voices_dir and cached there; the name
    # is a locale-qualified voice id from the rhasspy/piper-voices set.
    tts_voice: str = "es_ES-davefx-medium"
    voices_dir: str = "assets/voices"

    # How Piper is reached.
    #   "http"      separate service, the default. piper-tts is GPL-3.0-or-later,
    #               and running it out of process is aggregation rather than
    #               linking.
    #   "inprocess" imports piper directly. Convenient locally, and a licensing
    #               decision that must be made deliberately.
    tts_backend: str = "http"
    # 5050 rather than 5000: macOS binds 5000 for AirPlay Receiver.
    tts_url: str = "http://localhost:5050"

    # Vision sampling. Both conditions must hold before a frame is sent: at
    # least vision_interval_s since the last upload, and enough visual change
    # since the last uploaded frame. The interval bounds cost; the threshold
    # suppresses redundant spend below that ceiling.
    vision_interval_s: float = 4.0
    vision_motion_threshold: float = 6.0
    vision_timeout_s: float = 20.0  # measured ~8s for qwen2.5vl:3b; off the turn path

    # Rendered video track geometry.
    video_width: int = 512
    video_height: int = 512
    video_fps: int = 25

    database_url: str = "sqlite+aiosqlite:///./avatar.db"
    redis_url: str = "redis://localhost:6379/0"

    assets_dir: str = "assets"

    # Signs session cookies. The default is obviously not a secret and is
    # rejected by assert_production_ready; set SESSION_SECRET in deployment.
    session_secret: str = "dev-session-secret-not-for-production"

    # Object storage. "s3" covers AWS S3, Cloudflare R2, Backblaze and MinIO -
    # set s3_endpoint_url for anything that is not AWS.
    storage_backend: str = "local"
    storage_root: str = "assets/blobs"
    s3_bucket: str = ""
    s3_region: str = "auto"
    s3_endpoint_url: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""

    # Identity training. "local" fakes a run so the flow can be built without
    # a GPU account; "replicate" is pay-per-run hosted training.
    training_backend: str = "local"
    replicate_api_token: str = ""
    # Pinned by version rather than name: an upstream change must not silently
    # alter what a customer's likeness was trained with.
    replicate_trainer_version: str = ""

    # Session cookies must be Secure in deployment. False locally because
    # http://localhost is not a secure origin and the cookie would be dropped.
    cookies_secure: bool = False

    # Origins allowed to call the gateway. Listed explicitly rather than
    # wildcarded because these responses carry room tokens. The dev port is
    # pinned high to avoid colliding with whatever else is on 3000.
    web_origins: str = "http://localhost:3100,http://127.0.0.1:3100"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.web_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached singleton. Tests construct Settings directly to bypass the cache."""
    return Settings()
