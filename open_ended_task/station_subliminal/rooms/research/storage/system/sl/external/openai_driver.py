import asyncio
from typing import Any, Literal
from sl.llm.data_models import LLMResponse, Chat
from sl import config
from sl.llm.services import SampleCfg
from sl.utils import fn_utils


_client = None


def get_client() -> Any:
    raise RuntimeError("Remote OpenAI access is disabled in the public task package.")


@fn_utils.auto_retry_async([Exception], max_retry_attempts=5)
@fn_utils.max_concurrency_async(max_size=1000)
async def sample(model_id: str, input_chat: Chat, sample_cfg: SampleCfg) -> LLMResponse:
    raise RuntimeError("Remote OpenAI access is disabled in the public task package.")


async def batch_sample(
    model_id: str, input_chats: list[Chat], sample_cfgs: list[SampleCfg]
) -> list[LLMResponse]:
    raise RuntimeError("Remote OpenAI access is disabled in the public task package.")


async def upload_file(file_path: str, purpose: Literal["fine-tune"]) -> Any:
    raise RuntimeError("Remote OpenAI access is disabled in the public task package.")
