from typing import Any, Optional
from argparse import Namespace
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
import json
import os
import urllib.request

HOME = os.environ["HOME"]
API_TOKENS_DIR = os.path.join(HOME, ".pythia", "api_tokens")

def _load_api_token(key, domain):
    env_key = "{}_API_KEY".format(key)
    name = domain
    path = os.path.join(API_TOKENS_DIR, name)
    api_token = os.environ.get(env_key)
    if api_token is None:
        try:
            with open(path, "r") as api_token_file:
                api_token = api_token_file.read().strip()
        except:
            pass
    return api_token

DEEPSEEK_API_KEY   = _load_api_token("DEEPSEEK",   "deepseek.com")
HYPERBOLIC_API_KEY = _load_api_token("HYPERBOLIC", "hyperbolic.xyz")
OPENROUTER_API_KEY = _load_api_token("OPENROUTER", "openrouter.ai")
TOGETHER_API_KEY   = _load_api_token("TOGETHER",   "together.xyz")
XAI_API_KEY        = _load_api_token("XAI",        "x.ai")

@dataclass
class _ApproxOracleResponse:
    sample: dict = None
    think: str = None
    value: str = None
    data: Any = None
    t0: str = None
    t1: str = None

@dataclass
class ApproxOracleEndpoint:
    model: Optional[str]
    endpoint_model: str
    endpoint_max_new_tokens: int
    endpoint_api_url: str
    endpoint_api_token: str
    endpoint_api_protocol: str
    endpoint_api_rps_limit: Optional[int] = None
    endpoint_extra_params: Optional[dict] = None

    @classmethod
    def from_model(cls, model: str) -> Any:
        if model == "deepseek-r1-20250120":
            return cls.deepseek_r1_20250120()
        elif model == "\"deepseek-r1-20250120\"":
            return cls.deepseek_r1_20250120()
        elif model == "deepseek-v3-chat-20250324":
            return cls.deepseek_v3_chat_20250324()
            #return cls.hyperbolic_deepseek_v3_20241226()
            #return cls.together_deepseek_v3_20241226()
        elif model == "\"deepseek-v3-chat-20250324\"":
            return cls.deepseek_v3_chat_20250324()
        elif model == "deepseek-r1-20250120-hyperbolic":
            return cls.hyperbolic_deepseek_r1_20250120()
        elif model == "deepseek-v3-20250324-hyperbolic":
            return cls.hyperbolic_deepseek_v3_20250324()
        elif model == "deepseek-v3-20241226-hyperbolic":
            return cls.hyperbolic_deepseek_v3_20241226()
        elif model == "deepseek-r1-20250120-together":
            return cls.together_deepseek_r1_20250120()
        elif model == "deepseek-v3-20241226-together":
            return cls.together_deepseek_v3_20241226()
        elif model == "qwen-2.5-vl-72b-instruct-together":
            return cls.together_qwen_2_5_vl_72b_instruct()
        elif model == "qwq-32b-hyperbolic":
            return cls.hyperbolic_qwq_32b()
        elif model == "qwen-2.5-coder-32b-instruct-hyperbolic":
            return cls.hyperbolic_qwen_2_5_coder_32b_instruct()
        else:
            raise NotImplementedError

    @classmethod
    def deepseek(cls, **kwargs) -> Any:
        return cls(
            endpoint_api_url = "https://api.deepseek.com",
            endpoint_api_token = DEEPSEEK_API_KEY,
            endpoint_api_protocol = "deepseek",
            **kwargs,
        )

    @classmethod
    def deepseek_r1_20250120(cls) -> Any:
        return cls.deepseek(
            model = "deepseek-r1-20250120",
            endpoint_model = "deepseek-reasoner",
            endpoint_max_new_tokens = 8192,
        )

    @classmethod
    def deepseek_v3_chat_20250324(cls) -> Any:
        return cls.deepseek(
            model = "deepseek-v3-chat-20250324",
            endpoint_model = "deepseek-chat",
            endpoint_max_new_tokens = 8192,
        )

    @classmethod
    def hyperbolic(cls, **kwargs) -> Any:
        return cls(
            endpoint_api_url = "https://api.hyperbolic.xyz",
            endpoint_api_token = HYPERBOLIC_API_KEY,
            endpoint_api_protocol = "openai",
            **kwargs,
        )

    @classmethod
    def hyperbolic_deepseek_r1_20250120(cls) -> Any:
        return cls.hyperbolic(
            model = "deepseek-r1-20250120-hyperbolic",
            endpoint_model = "deepseek-ai/DeepSeek-R1",
            endpoint_max_new_tokens = 4096,
        )

    @classmethod
    def hyperbolic_deepseek_r1_zero_20250120(cls) -> Any:
        return cls.hyperbolic(
            model = "deepseek-r1-zero-20250120-hyperbolic",
            endpoint_model = "deepseek-ai/DeepSeek-R1-Zero",
            endpoint_max_new_tokens = 4096,
        )

    @classmethod
    def hyperbolic_deepseek_v3_20250324(cls) -> Any:
        return cls.hyperbolic(
            model = "deepseek-v3-20250324-hyperbolic",
            endpoint_model = "deepseek-ai/DeepSeek-V3-0324",
            endpoint_max_new_tokens = 4096,
        )

    @classmethod
    def hyperbolic_deepseek_v3_20241226(cls) -> Any:
        return cls.hyperbolic(
            model = "deepseek-v3-20241226-hyperbolic",
            endpoint_model = "deepseek-ai/DeepSeek-V3",
            endpoint_max_new_tokens = 4096,
        )

    @classmethod
    def hyperbolic_qwq_32b(cls) -> Any:
        return cls.hyperbolic(
            model = "qwq-32b-hyperbolic",
            endpoint_model = "Qwen/QwQ-32B",
            endpoint_max_new_tokens = 32768,
        )

    @classmethod
    def hyperbolic_qwen_2_5_coder_32b_instruct(cls) -> Any:
        return cls.hyperbolic(
            model = "qwen-2.5-coder-32b-instruct-hyperbolic",
            endpoint_model = "Qwen/Qwen2.5-Coder-32B-Instruct",
            endpoint_max_new_tokens = 8192,
        )

    @classmethod
    def hyperbolic_llama_3_1_405b_instruct(cls) -> Any:
        return cls.hyperbolic(
            model = "hyperbolic-llama-3.1-405b-instruct",
            endpoint_model = "meta-llama/Meta-Llama-3.1-405B-Instruct",
            endpoint_max_new_tokens = 4096,
        )

    @classmethod
    def hyperbolic_llama_3_1_405b_base_bf16(cls) -> Any:
        return cls.hyperbolic(
            model = "hyperbolic-llama-3.1-405b-base",
            endpoint_model = "meta-llama/Meta-Llama-3.1-405B",
            endpoint_max_new_tokens = 4096,
        )

    @classmethod
    def hyperbolic_llama_3_1_405b_base_fp8(cls) -> Any:
        return cls.hyperbolic(
            model = "hyperbolic-llama-3.1-405b-base-fp8",
            endpoint_model = "meta-llama/Meta-Llama-3.1-405B-FP8",
            endpoint_max_new_tokens = 4096,
        )

    @classmethod
    def openrouter(cls, **kwargs) -> Any:
        return cls(
            endpoint_api_url = "https://openrouter.ai/api",
            endpoint_api_token = OPENROUTER_API_KEY,
            endpoint_api_protocol = "openai",
            **kwargs,
        )

    @classmethod
    def together(cls, **kwargs) -> Any:
        return cls(
            endpoint_api_url = "https://api.together.xyz/v1",
            endpoint_api_token = TOGETHER_API_KEY,
            endpoint_api_protocol = "openai",
            **kwargs,
        )

    @classmethod
    def together_deepseek_r1_20250120(cls) -> Any:
        return cls.together(
            model = "deepseek-r1-20250120-together",
            endpoint_model = "deepseek-ai/DeepSeek-R1",
            endpoint_max_new_tokens = 32768,
        )

    @classmethod
    def together_deepseek_v3_20241226(cls) -> Any:
        return cls.together(
            model = "deepseek-v3-20241226-together",
            endpoint_model = "deepseek-ai/DeepSeek-V3",
            endpoint_max_new_tokens = 16384,
        )

    @classmethod
    def together_qwen_2_5_vl_72b_instruct(cls) -> Any:
        return cls.together(
            model = "qwen-2.5-vl-72b-instruct-together",
            endpoint_model = "Qwen/Qwen2.5-VL-72B-Instruct",
            endpoint_max_new_tokens = 16384,
        )

    @classmethod
    def xai(cls, **kwargs) -> Any:
        return cls(
            endpoint_api_url = "https://api.x.ai",
            endpoint_api_token = XAI_API_KEY,
            endpoint_api_protocol = "openai",
            **kwargs,
        )

    @classmethod
    def xai_grok_3_mini_beta(cls) -> Any:
        return cls.xai(
            model = "xai-grok-3-mini-beta-20250418",
            endpoint_model = "grok-3-mini-beta",
            endpoint_max_new_tokens = 131072,
            endpoint_extra_params = {
                "reasoning_effort": "high",
            },
        )

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = self.endpoint_model
        if (
            self.endpoint_api_protocol == "deepseek" or
            self.endpoint_api_protocol == "openai"
        ):
            # TODO: proper urllib formatting.
            if self.endpoint_api_protocol == "openai":
                self._chat_endpoint_url = "{}/v1/chat/completions".format(self.endpoint_api_url)
            else:
                self._chat_endpoint_url = "{}/chat/completions".format(self.endpoint_api_url)
            self._chat_endpoint_headers = {
                "User-Agent": "curl/8.7.1",
                "Authorization": "Bearer {}".format(self.endpoint_api_token),
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        else:
            raise NotImplementedError

    def query(
        self,
        messages: list[dict[str, str]],
        # FIXME: unused sampling params.
        sample: dict[str, Any] = None,
    ) -> _ApproxOracleResponse:
        if (
            self.endpoint_api_protocol == "deepseek" or
            self.endpoint_api_protocol == "openai"
        ):
            req_body = {
                "messages": messages,
                "model": self.endpoint_model,
                "stream": False,
            }
            if self.endpoint_api_protocol == "deepseek":
                req_body["max_new_tokens"] = self.endpoint_max_new_tokens
            elif self.endpoint_api_protocol == "openai":
                req_body["max_tokens"] = self.endpoint_max_new_tokens
            if (
                self.endpoint_api_protocol == "deepseek" and
                self.endpoint_model == "deepseek-reasoner"
            ):
                pass
            elif (
                self.endpoint_api_protocol == "deepseek" and
                self.endpoint_model == "deepseek-chat"
            ):
                req_body |= {
                    # TODO: configure sampling params.
                    "temperature": 0,
                }
            elif (
                self.endpoint_api_url.startswith("https://api.hyperbolic.xyz") and
                self.endpoint_model.startswith("deepseek-ai/")
            ):
                req_body |= {
                    # TODO: configure sampling params.
                    "temperature": 0,
                }
            else:
                req_body |= {
                    # TODO: configure sampling params.
                    "temperature": 0,
                    "top_p": 1,
                    #"logprobs": True,
                }
        else:
            raise NotImplementedError
        if self.endpoint_extra_params is not None:
            req_body |= self.endpoint_extra_params
        req_data = json.dumps(req_body).encode("utf-8")
        #print(f"DEBUG: ApproxOracleEndpoint.query: req data = {req_data}")
        req = urllib.request.Request(
            self._chat_endpoint_url,
            headers = self._chat_endpoint_headers.copy(),
            data = req_data,
        )
        t0 = datetime.utcnow()
        with urllib.request.urlopen(req) as res:
            res_data = res.read()
        t1 = datetime.utcnow()
        res_body = json.loads(res_data.decode("utf-8"))
        #print(f"DEBUG: ApproxOracleEndpoint.query: res body = {res_body}")
        think = None
        if (
            self.endpoint_api_protocol == "deepseek" or
            self.endpoint_api_protocol == "openai"
        ):
            think = res_body["choices"][0]["message"].pop("reasoning_content", None)
            value = res_body["choices"][0]["message"].pop("content", None)
            if think is None and value.startswith("<think>\n"):
                think_end_pos = value.rfind("</think>\n\n")
                if think_end_pos >= 0:
                    think = value[8:think_end_pos]
                    value = value[think_end_pos+10:]
        else:
            raise NotImplementedError
        return _ApproxOracleResponse(
            sample={
                "temperature": req_body.get("temperature", None),
            },
            think=think,
            value=value,
            data=json.dumps(res_body),
            t0=t0.isoformat(),
            t1=t1.isoformat(),
        )

@dataclass
class ApproxOracleItem:
    #timestamp: str = None
    key: Any = None
    query: str = None
    model: str = None
    sample: dict = None
    think: str = None
    value: str = None
    extra: Any = None

@dataclass
class ApproxOracleExtraItem:
    res: Any = None

@dataclass
class ApproxOracleResExtra:
    data: str = None
    t0: str = None
    t1: str = None

@dataclass
class ApproxOracleTestItem:
    timestamp: str = None
    model: str = None

@dataclass
class _ApproxOracleWorkItem:
    item: ApproxOracleItem
    res: _ApproxOracleResponse = None

def _query(work_item):
    print(f"DEBUG: _query: work item = {work_item}")
    endpoint = ApproxOracleEndpoint.from_model(work_item.item.model)
    print(f"DEBUG: _query: endpoint.model = {endpoint.model}")
    messages = [
        {
            "role": "user",
            "content": work_item.item.query,
        }
    ]
    work_item.res = endpoint.query(messages)
    print(f"DEBUG: _query: done")
    return work_item

def _try_query(work_item):
    print(f"DEBUG: _try_query: work item = {work_item}")
    try:
        result = _query(work_item)
    # TODO: exc reporting.
    except Exception as e:
        print(f"DEBUG: _try_query: except = {e}")
        work_item.res = None
        result = work_item
    return result

@dataclass
class ApproxOracleInterface_:
    work_rx: Any
    result_tx: Any

    def __post_init__(self) -> None:
        pass

@dataclass
class ApproxOracleInterface:
    default_model: str = "deepseek-v3-chat-20250324"
    default_timeout: int = 480
    concurrency: int = 64

    def __post_init__(self) -> None:
        print(f"DEBUG: ApproxOracleInterface.__post_init__")
        self._poolexec = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.concurrency,
        )
        self._work_set = set()
        self._done_set = set()

    def __len__(self) -> int:
        return len(self._work_set)

    def put(self, item: ApproxOracleItem | dict) -> None:
        if isinstance(item, dict):
            #print("DEBUG: ApproxOracleInterface.put: isa dict")
            item = ApproxOracleItem(**item)
        if item.model is None:
            #print("DEBUG: ApproxOracleInterface.put: set default model")
            item.model = self.default_model
        print(f"DEBUG: ApproxOracleInterface.put: item = {item}")
        #return
        work_item = _ApproxOracleWorkItem(item)
        w = self._poolexec.submit(_try_query, work_item)
        self._work_set.add(w)

    def poll(self, timeout=None) -> ApproxOracleItem:
        print(f"DEBUG: ApproxOracleInterface.poll")
        if timeout is None:
            timeout = self.default_timeout
        if not self._done_set:
            try:
                done, rem = concurrent.futures.wait(
                    self._work_set,
                    timeout=timeout,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                self._done_set = done
                self._work_set = rem
            except TimeoutError:
                pass
        if not self._done_set:
            return None
        print(f"DEBUG: ApproxOracleInterface.poll: completed")
        w = self._done_set.pop()
        work_item = w.result()
        item = work_item.item
        if work_item.res is None:
            print(f"DEBUG: ApproxOracleInterface.poll: no res")
            return item
        # FIXME
        item.sample = work_item.res.sample
        item.think = work_item.res.think
        item.value = work_item.res.value
        #item.timestamp = work_item.res.t1
        if False:
            item.extra = {
                "res.data": work_item.res.data,
                "res.t0": work_item.res.t0,
                "res.t1": work_item.res.t1,
            }
        if True:
            item.extra = ApproxOracleExtraItem(
                res = ApproxOracleResExtra(
                    data = work_item.res.data,
                    t0 = work_item.res.t0,
                    t1 = work_item.res.t1,
                )
            )
        print(f"DEBUG: ApproxOracleInterface.poll: item = {item}")
        return item

    def poll_test(self, timeout=None) -> ApproxOracleTestItem:
        return ApproxOracleTestItem(
            timestamp = datetime.utcnow().isoformat(),
            model = self.default_model,
        )

@dataclass
class ApproxOracleAsyncInterface(ApproxOracleInterface):
    pass
