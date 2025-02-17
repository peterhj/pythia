from typing import Any, Dict, List, Optional, Tuple
from argparse import Namespace
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
import json
import os
import urllib.request

HOME = os.environ["HOME"]
API_KEYS_DIR = os.path.join(HOME, ".pythia", "api_keys")

def _load_api_key(key, domain):
    env_key = "{}_API_KEY".format(key)
    name = domain
    path = os.path.join(API_KEYS_DIR, name)
    api_key = os.environ.get(env_key)
    if api_key is None:
        try:
            with open(path, "r") as api_key_file:
                api_key = api_key_file.read().strip()
        except:
            pass
    return api_key

DEEPSEEK_API_KEY = _load_api_key("DEEPSEEK", "deepseek.com")
HYPERBOLIC_API_KEY = _load_api_key("HYPERBOLIC", "hyperbolic.xyz")
TOGETHER_API_KEY = _load_api_key("TOGETHER", "together.xyz")

@dataclass
class _ApproxOracleResponse:
    thinking: str = None
    value: str = None
    data: Any = None
    t0: str = None
    t1: str = None

@dataclass
class ApproxOracleEndpoint:
    model: Optional[str]
    endpoint_model: str
    endpoint_max_tokens: int
    endpoint_api_url: str
    endpoint_api_key: str
    endpoint_protocol: str

    @classmethod
    def from_model(cls, model: str) -> Any:
        if model == "deepseek-r1-20250120":
            return cls.deepseek_r1_20250120()
        elif model == "deepseek-v3-chat-20241226":
            #return cls.deepseek_v3_chat_20241226()
            #return cls.hyperbolic_deepseek_v3_20241226()
            return cls.together_deepseek_v3_20241226()
        elif model == "deepseek-r1-20250120-hyperbolic":
            return cls.hyperbolic_deepseek_r1_20250120()
        elif model == "deepseek-v3-20241226-hyperbolic":
            return cls.hyperbolic_deepseek_v3_20241226()
        elif model == "deepseek-r1-20250120-together":
            return cls.together_deepseek_r1_20250120()
        elif model == "deepseek-v3-20241226-together":
            return cls.together_deepseek_v3_20241226()
        else:
            raise NotImplementedError

    @classmethod
    def deepseek(cls, **kwargs) -> Any:
        return cls(
            endpoint_api_url = "https://api.deepseek.com",
            endpoint_api_key = DEEPSEEK_API_KEY,
            endpoint_protocol = "deepseek",
            **kwargs,
        )

    @classmethod
    def deepseek_r1_20250120(cls) -> Any:
        return cls.deepseek(
            model = "deepseek-r1-20250120",
            endpoint_model = "deepseek-reasoner",
            endpoint_max_tokens = 8192,
        )

    @classmethod
    def deepseek_v3_chat_20241226(cls) -> Any:
        return cls.deepseek(
            model = "deepseek-v3-chat-20241226",
            endpoint_model = "deepseek-chat",
            endpoint_max_tokens = 8192,
        )

    @classmethod
    def hyperbolic(cls, **kwargs) -> Any:
        return cls(
            endpoint_api_url = "https://api.hyperbolic.xyz",
            endpoint_api_key = HYPERBOLIC_API_KEY,
            endpoint_protocol = "openai",
            **kwargs,
        )

    @classmethod
    def hyperbolic_deepseek_r1_20250120(cls) -> Any:
        return cls.hyperbolic(
            model = "deepseek-r1-20250120-hyperbolic",
            endpoint_model = "deepseek-ai/DeepSeek-R1",
            endpoint_max_tokens = 4096,
        )

    @classmethod
    def hyperbolic_deepseek_r1_zero_20250120(cls) -> Any:
        return cls.hyperbolic(
            model = "deepseek-r1-zero-20250120-hyperbolic",
            endpoint_model = "deepseek-ai/DeepSeek-R1-Zero",
            endpoint_max_tokens = 4096,
        )

    @classmethod
    def hyperbolic_deepseek_v3_20241226(cls) -> Any:
        return cls.hyperbolic(
            model = "deepseek-v3-20241226-hyperbolic",
            endpoint_model = "deepseek-ai/DeepSeek-V3",
            endpoint_max_tokens = 4096,
        )

    @classmethod
    def hyperbolic_llama_3_1_405b_instruct(cls) -> Any:
        return cls.hyperbolic(
            model = "hyperbolic-llama-3.1-405b-instruct",
            endpoint_model = "meta-llama/Meta-Llama-3.1-405B-Instruct",
            endpoint_max_tokens = 4096,
        )

    @classmethod
    def hyperbolic_llama_3_1_405b_base_bf16(cls) -> Any:
        return cls.hyperbolic(
            model = "hyperbolic-llama-3.1-405b-base",
            endpoint_model = "meta-llama/Meta-Llama-3.1-405B",
            endpoint_max_tokens = 4096,
        )

    @classmethod
    def hyperbolic_llama_3_1_405b_base_fp8(cls) -> Any:
        return cls.hyperbolic(
            model = "hyperbolic-llama-3.1-405b-base-fp8",
            endpoint_model = "meta-llama/Meta-Llama-3.1-405B-FP8",
            endpoint_max_tokens = 4096,
        )

    @classmethod
    def together(cls, **kwargs) -> Any:
        return cls(
            endpoint_api_url = "https://api.together.xyz/v1",
            endpoint_api_key = TOGETHER_API_KEY,
            endpoint_protocol = "openai",
            **kwargs,
        )

    @classmethod
    def together_deepseek_r1_20250120(cls) -> Any:
        return cls.together(
            model = "deepseek-r1-20250120-together",
            endpoint_model = "deepseek-ai/DeepSeek-R1",
            endpoint_max_tokens = 32768,
        )

    @classmethod
    def together_deepseek_v3_20241226(cls) -> Any:
        return cls.together(
            model = "deepseek-v3-20241226-together",
            endpoint_model = "deepseek-ai/DeepSeek-V3",
            endpoint_max_tokens = 16384,
        )

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = self.endpoint_model
        if (
            self.endpoint_protocol == "deepseek" or
            self.endpoint_protocol == "openai"
        ):
            # TODO: proper urllib formatting.
            if self.endpoint_protocol == "openai":
                self._chat_endpoint_url = "{}/v1/chat/completions".format(self.endpoint_api_url)
            else:
                self._chat_endpoint_url = "{}/chat/completions".format(self.endpoint_api_url)
            self._chat_endpoint_headers = {
                "User-Agent": "curl/8.7.1",
                "Authorization": "Bearer {}".format(self.endpoint_api_key),
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        else:
            raise NotImplementedError

    def query(self, messages: List[Dict[str, str]], *args) -> _ApproxOracleResponse:
        if (
            self.endpoint_protocol == "deepseek" or
            self.endpoint_protocol == "openai"
        ):
            req_body = {
                "messages": messages,
                "model": self.endpoint_model,
                "stream": False,
                "max_tokens": self.endpoint_max_tokens,
            }
            if (
                self.endpoint_protocol == "deepseek" and
                self.endpoint_model == "deepseek-reasoner"
            ):
                pass
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
                    "logprobs": True,
                }
        else:
            raise NotImplementedError
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
        thinking = None
        if (
            self.endpoint_protocol == "deepseek" or
            self.endpoint_protocol == "openai"
        ):
            value = res_body["choices"][0]["message"]["content"]
            if "reasoning_content" in res_body["choices"][0]["message"]:
                thinking = res_body["choices"][0]["message"]["reasoning_content"]
            elif value.startswith("<think>\n"):
                think_end_pos = value.rfind("</think>\n\n")
                if think_end_pos >= 0:
                    thinking = value[8:think_end_pos]
                    value = value[think_end_pos+10:]
        else:
            raise NotImplementedError
        return _ApproxOracleResponse(
            thinking=thinking,
            value=value,
            data=json.dumps(res_body),
            t0=t0.isoformat(),
            t1=t1.isoformat(),
        )

@dataclass
class ApproxOracleItem:
    key: Any
    query: str
    model: str = None
    thinking: str = None
    value: str = None
    timestamp: str = None
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
class ApproxOracleInterface:
    default_model: str = "deepseek-v3-chat-20241226"
    default_timeout: int = 480
    concurrency: int = 64

    def __post_init__(self) -> None:
        print(f"DEBUG: ApproxOracleInterface.__post_init__")
        self._poolexec = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.concurrency,
        )
        self._work_set = set()

    def __len__(self) -> int:
        return len(self._work_set)

    def put(self, item: ApproxOracleItem) -> None:
        if isinstance(item, dict):
            print("DEBUG: ApproxOracleInterface.put: isa dict")
            item = ApproxOracleItem(**item)
        if item.model is None:
            print("DEBUG: ApproxOracleInterface.put: set default model")
            item.model = self.default_model
        print(f"DEBUG: ApproxOracleInterface.put: item = {item}")
        #return
        work_item = _ApproxOracleWorkItem(item)
        w = self._poolexec.submit(_try_query, work_item)
        self._work_set.add(w)

    def get(self, timeout=None) -> ApproxOracleItem:
        print(f"DEBUG: ApproxOracleInterface.get")
        if timeout is None:
            timeout = self.default_timeout
        try:
            for w in concurrent.futures.as_completed(
                self._work_set,
                timeout=timeout
            ):
                print(f"DEBUG: ApproxOracleInterface.get: completed")
                work_item = w.result()
                item = work_item.item
                if work_item.res is None:
                    print(f"DEBUG: ApproxOracleInterface.get: no res")
                    return item
                item.thinking = work_item.res.thinking
                item.value = work_item.res.value
                item.timestamp = work_item.res.t1
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
                print(f"DEBUG: ApproxOracleInterface.get: item = {item}")
                return item
        except TimeoutError:
            pass
        return None

    def get_test(self, timeout=None) -> ApproxOracleTestItem:
        return ApproxOracleTestItem(
            timestamp = datetime.utcnow().isoformat(),
            model = self.default_model,
        )
