from typing import Any, Optional, Union
from argparse import Namespace
import asyncio
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os
import threading
import time
import traceback
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

def _match_str(query: str, pat: str) -> bool:
    return query == pat or query == f"\"{pat}\""

@dataclass
class _ApproxOracleResponseItem:
    sample: dict = None
    think: str = None
    value: str = None
    data: Any = None
    t0: str = None
    t1: str = None

@dataclass
class ApproxOracleExceptItem:
    exc_type: str = None
    exc_str: str = None
    stack_trace: str = None

@dataclass
class ApproxOracleEndpoint:
    model: Optional[str]
    endpoint_model: str
    endpoint_max_new_tokens: int
    endpoint_api_url: str
    endpoint_api_token: str
    endpoint_api_protocol: str
    endpoint_extra_params: Optional[dict] = None
    endpoint_throttle_rps: Optional[int] = None

    @classmethod
    def from_model(cls, model: str) -> Any:
        if _match_str(model, "deepseek-r1-20250120"):
            return cls.deepseek_r1_20250120()
        elif _match_str(model, "deepseek-v3-chat-20250324"):
            return cls.deepseek_v3_chat_20250324()
        elif _match_str(model, "xai-grok-3-mini-20250520"):
            return cls.xai_grok_3_mini()
        elif _match_str(model, "xai-grok-3-mini-beta-20250418"):
            return cls.xai_grok_3_mini_beta()
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
            endpoint_throttle_rps = 64,
        )

    @classmethod
    def deepseek_v3_chat_20250324(cls) -> Any:
        return cls.deepseek(
            model = "deepseek-v3-chat-20250324",
            endpoint_model = "deepseek-chat",
            endpoint_max_new_tokens = 8192,
            endpoint_throttle_rps = 64,
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
            endpoint_api_url = "https://openrouter.ai",
            endpoint_api_token = OPENROUTER_API_KEY,
            endpoint_api_protocol = "openrouter",
            **kwargs,
        )

    @classmethod
    def openrouter_deepseek_r1_20250120(cls) -> Any:
        return cls.openrouter(
            model = "deepseek-r1-20250120-openrouter",
            endpoint_model = "deepseek/deepseek-r1",
            endpoint_max_new_tokens = 8192,
        )

    @classmethod
    def openrouter_deepseek_v3_20250324(cls) -> Any:
        return cls.openrouter(
            model = "deepseek-v3-20250324-openrouter",
            endpoint_model = "deepseek/deepseek-chat-v3-0324",
            endpoint_max_new_tokens = 8192,
        )

    @classmethod
    def openrouter_grok_3_mini_beta(cls) -> Any:
        return cls.openrouter(
            model = "xai-grok-3-mini-beta-20250418-openrouter",
            endpoint_model = "x-ai/grok-3-mini-beta",
            endpoint_max_new_tokens = 131072,
        )

    @classmethod
    def together(cls, **kwargs) -> Any:
        return cls(
            endpoint_api_url = "https://api.together.xyz",
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
    def xai_grok_3_mini(cls) -> Any:
        return cls.xai(
            model = "xai-grok-3-mini-20250520",
            endpoint_model = "grok-3-mini",
            endpoint_max_new_tokens = 131072,
            endpoint_extra_params = {
                "reasoning_effort": "high",
            },
            endpoint_throttle_rps = 10,
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
            endpoint_throttle_rps = 10,
        )

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = self.endpoint_model
        if self.endpoint_api_protocol in (
            "deepseek",
            "openai",
            "openrouter",
        ):
            # TODO: proper urllib formatting.
            if self.endpoint_api_protocol == "deepseek":
                self._chat_endpoint_url = "{}/chat/completions".format(self.endpoint_api_url)
            elif self.endpoint_api_protocol == "openai":
                self._chat_endpoint_url = "{}/v1/chat/completions".format(self.endpoint_api_url)
            elif self.endpoint_api_protocol == "openrouter":
                self._chat_endpoint_url = "{}/api/v1/chat/completions".format(self.endpoint_api_url)
            else:
                raise NotImplementedError
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
        messages: list[dict[str, str]] = None,
        sample: dict[str, Any] = None,
        res: _ApproxOracleResponseItem = None,
        key: Any = None,
    ) -> _ApproxOracleResponseItem:
        if res is None:
            res = _ApproxOracleResponseItem()
        if self.endpoint_api_protocol in (
            "deepseek",
            "openai",
            "openrouter",
        ):
            req_body = dict()
            req_body["model"] = self.endpoint_model
            if self.endpoint_api_protocol == "openrouter":
                req_body["models"] = []
            req_body["messages"] = messages
            req_body["stream"] = False
            if self.endpoint_api_protocol == "deepseek":
                req_body["max_new_tokens"] = self.endpoint_max_new_tokens
            else:
                req_body["max_tokens"] = self.endpoint_max_new_tokens
            # NB: default sampling params.
            if (
                self.endpoint_api_protocol == "deepseek" and
                self.endpoint_model == "deepseek-reasoner"
            ):
                pass
            elif (
                self.endpoint_api_protocol == "deepseek" and
                self.endpoint_model == "deepseek-chat"
            ):
                if sample is None:
                    sample = dict()
                if sample.get("temperature", None) is None:
                    sample["temperature"] = 0.0
            elif (
                self.endpoint_api_url.startswith("https://api.hyperbolic.xyz") and
                self.endpoint_model.startswith("deepseek-ai/")
            ):
                if sample is None:
                    sample = dict()
                if sample.get("temperature", None) is None:
                    sample["temperature"] = 0.0
            elif self.model == "xai-grok-3-mini-beta-20250418":
                pass
            else:
                if sample is None:
                    sample = dict()
                if sample.get("temperature", None) is None:
                    sample["temperature"] = 0.0
                if sample.get("top_p", None) is None:
                    sample["top_p"] = 1.0
        else:
            raise NotImplementedError
        if sample is not None:
            res.sample = sample
            req_body |= sample
        if self.endpoint_extra_params is not None:
            req_body |= self.endpoint_extra_params
        temp = req_body.get("temperature", None)
        #print(f"DEBUG: ApproxOracleEndpoint.query: req body = {req_body}")
        req_data = json.dumps(req_body).encode("utf-8")
        #print(f"DEBUG: ApproxOracleEndpoint.query: url      = {self._chat_endpoint_url}")
        #print(f"DEBUG: ApproxOracleEndpoint.query: headers  = {self._chat_endpoint_headers}")
        #print(f"DEBUG: ApproxOracleEndpoint.query: req data = {req_data}")
        hreq = urllib.request.Request(
            self._chat_endpoint_url,
            headers = self._chat_endpoint_headers.copy(),
            data = req_data,
        )
        res.t0 = datetime.utcnow().isoformat()
        #print(f"DEBUG: ApproxOracleEndpoint.query: t0 = {res.t0}")
        with urllib.request.urlopen(hreq) as hres:
            res_data = hres.read()
        res.t1 = datetime.utcnow().isoformat()
        #print(f"DEBUG: ApproxOracleEndpoint.query: t1 = {res.t1}")
        #print(f"DEBUG: ApproxOracleEndpoint.query: recv data = {json.dumps(res_data.decode('utf-8'))}", flush=True)
        res_body = json.loads(res_data.decode("utf-8"))
        #print(f"DEBUG: ApproxOracleEndpoint.query: res body = {res_body}")
        think = None
        value = None
        if self.endpoint_api_protocol in (
            "deepseek",
            "openai",
            "openrouter",
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
        res.think = think
        res.value = value
        # NB: re-serializing json response.
        res.data = json.dumps(res_body)
        if key is not None:
            print(f"DEBUG: ApproxOracleEndpoint.query: done: key = {key}")
        else:
            print(f"DEBUG: ApproxOracleEndpoint.query: done")
        return res

@dataclass
class ApproxOracleSampleItem:
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None

@dataclass
class ApproxOracleItem:
    #timestamp: str = None
    key: Any = None
    query: str = None
    model: str = None
    ctr: int = None
    sample: ApproxOracleSampleItem = None
    think: str = None
    value: str = None
    extra: Any = None

@dataclass
class ApproxOracleResExtra:
    data: str = None
    t0: str = None
    t1: str = None

@dataclass
class ApproxOracleExtraItem:
    res: ApproxOracleResExtra = None
    exc: ApproxOracleExceptItem = None

@dataclass
class ApproxOracleTestItem:
    timestamp: str = None
    model: str = None

@dataclass
class _ApproxOracleWorkItem:
    item: ApproxOracleItem
    res: _ApproxOracleResponseItem = None
    exc: ApproxOracleExceptItem = None

    def _finalize(self) -> ApproxOracleItem:
        item = self.item
        if self.res is None:
            #print(f"DEBUG: _ApproxOracleWorkItem._finalize: no res")
            item.extra = ApproxOracleExtraItem(
                exc=self.exc,
            )
            return item
        if self.res.sample is not None:
            item.sample = ApproxOracleSampleItem(**self.res.sample)
        item.think = self.res.think
        item.value = self.res.value
        item.extra = ApproxOracleExtraItem(
            res=ApproxOracleResExtra(
                data=self.res.data,
                t0=self.res.t0,
                t1=self.res.t1,
            ),
            exc=self.exc,
        )
        #print(f"DEBUG: _ApproxOracleWorkItem._finalize: item = {item}")
        return item

def _query(work_item, key: Any = None):
    #print(f"DEBUG: _query: pre work item  = {work_item}")
    endpoint = ApproxOracleEndpoint.from_model(work_item.item.model)
    #print(f"DEBUG: _query: endpoint.model = {endpoint.model}")
    messages = [
        {
            "role": "user",
            "content": work_item.item.query,
        }
    ]
    work_item.res = _ApproxOracleResponseItem()
    endpoint.query(
        messages=messages,
        res=work_item.res,
        key=key,
    )
    #print(f"DEBUG: _query: post work item = {work_item}")
    #print(f"DEBUG: _query: done")
    return work_item

def _try_query(work_item, key: Any = None):
    #print(f"DEBUG: _try_query: work item = {work_item}")
    try:
        _query(work_item, key=key)
    # TODO: exc reporting.
    except Exception as e:
        #print(f"DEBUG: _try_query: except = {e}")
        work_item.exc = ApproxOracleExceptItem(
            exc_type=f"{type(e).__name__}",
            exc_str=str(e),
            stack_trace=traceback.format_exc(),
        )
        print(f"DEBUG: _try_query: except = {work_item.exc}")
    return work_item

@dataclass
class ApproxOracleWorker:
    concurrency: int
    _poolexec: concurrent.futures.ThreadPoolExecutor = None

    def __post_init__(self) -> None:
        print(f"DEBUG: ApproxOracleWorker.__post_init__")
        if self._poolexec is None:
            self._poolexec = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.concurrency,
            )

@dataclass
class ApproxOracleInterface_:
    work_rx: Any
    result_tx: Any

    def __post_init__(self) -> None:
        pass

@dataclass
class ApproxOracleInterface:
    worker: ApproxOracleWorker = None
    default_model: str = "deepseek-v3-chat-20250324"
    default_timeout: int = 1620
    concurrency: int = 192

    def __post_init__(self) -> None:
        print(f"DEBUG: ApproxOracleInterface.__post_init__")
        if self.worker is None:
            self.worker = ApproxOracleWorker(self.concurrency)
        self._work_set = set()
        self._done_set = set()

    def __len__(self) -> int:
        return len(self._work_set)

    def put(self, item: Union[ApproxOracleItem, dict]) -> None:
        if isinstance(item, dict):
            #print("DEBUG: ApproxOracleInterface.put: isa dict")
            item = ApproxOracleItem(**item)
        if item.model is None or _match_str(item.model, "default"):
            #print("DEBUG: ApproxOracleInterface.put: set default model")
            item.model = self.default_model
        print(f"DEBUG: ApproxOracleInterface.put: item = {item}")
        work_item = _ApproxOracleWorkItem(item)
        w = self.worker._poolexec.submit(_try_query, work_item, key=item.key)
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
        if False:
            item = work_item.item
            if work_item.res is None:
                print(f"DEBUG: ApproxOracleInterface.poll: no res")
                return item
            # FIXME
            item.sample = work_item.res.sample
            item.think = work_item.res.think
            item.value = work_item.res.value
            #item.timestamp = work_item.res.t1
            item.extra = ApproxOracleExtraItem(
                res = ApproxOracleResExtra(
                    data = work_item.res.data,
                    t0 = work_item.res.t0,
                    t1 = work_item.res.t1,
                )
            )
            print(f"DEBUG: ApproxOracleInterface.poll: item = {item}")
            return item
        return work_item._finalize()

    def poll_test(self, timeout=None) -> ApproxOracleTestItem:
        return ApproxOracleTestItem(
            timestamp = datetime.utcnow().isoformat(),
            model = self.default_model,
        )

@dataclass
class ApproxOracleAsyncInterface:
    worker: ApproxOracleWorker = None
    default_model: str = "deepseek-v3-chat-20250324"
    default_timeout: int = 1620
    concurrency: int = 192
    shutdown_t1: str = None

    _get_lock: Any = None
    _next_get_t0: Any = None

    def __post_init__(self) -> None:
        print(f"DEBUG: ApproxOracleAsyncInterface.__post_init__")
        if self.worker is None:
            self.worker = ApproxOracleWorker(self.concurrency)
        self._get_lock = threading.Lock()

    async def get(self, item: Union[ApproxOracleItem, dict]) -> ApproxOracleItem:
        if isinstance(item, dict):
            item = ApproxOracleItem(**item)
        if item.model is None or _match_str(item.model, "default"):
            item.model = self.default_model
        endpoint = ApproxOracleEndpoint.from_model(item.model)
        work_item = _ApproxOracleWorkItem(item)
        def _query_work_item():
            t = datetime.utcnow()
            t0 = None
            if self.shutdown_t1 is not None:
                try:
                    t1 = datetime.fromisoformat(self.shutdown_t1)
                    if t >= t1:
                        print(f"DEBUG: ApproxOracleAsyncInterface.get: key = {item.key} t = {t.isoformat()} t1 = {t1.isoformat()} shutdown")
                        return work_item
                except Exception as e:
                    print(f"DEBUG: ApproxOracleAsyncInterface.get: key = {item.key} t = {t.isoformat()} shutdown except = {e}")
                    return work_item
            if endpoint.endpoint_throttle_rps is not None:
                throttle_delay = 1.0 / endpoint.endpoint_throttle_rps
            else:
                throttle_delay = None
            if throttle_delay is not None:
                delta_t = timedelta(seconds=throttle_delay)
                with self._get_lock:
                    t0 = self._next_get_t0
                    if t0 is not None:
                        self._next_get_t0 = max(t0, t) + delta_t
                    else:
                        self._next_get_t0 = t + delta_t
                while t0 is not None and t0 > t:
                    time.sleep((t0 - t).total_seconds())
                    t = datetime.utcnow()
            print(f"DEBUG: ApproxOracleAsyncInterface.get: key = {item.key} t = {t.isoformat()} t0 = {t0.isoformat() if t0 is not None else None}")
            _try_query(work_item, key=item.key)
            return work_item
        loop = asyncio.get_running_loop()
        w = loop.run_in_executor(self.worker._poolexec, _query_work_item)
        work_item = await w
        return work_item._finalize()

    #def get_sync(self, item: Union[ApproxOracleItem, dict]):

def test_main():
    print(f"DEBUG: _approx_oracle: test main")
    iface = ApproxOracleInterface(
        #default_model="deepseek-r1-20250120",
    )
    item = ApproxOracleItem(
        key=0,
        query="""How would I use `asyncio.wrap_future` with both a `concurrent.futures.ThreadPoolExecutor` and an asyncio loop? Please provide a full toy example that involves making a POST request (using `urllib.request`) to "http://api.example.com/".""",
    )
    iface.put(item)
    result = iface.poll()
    print(result)

def test_main_async():
    print(f"DEBUG: _approx_oracle: test main (async)")
    iface = ApproxOracleAsyncInterface(
        #default_model="deepseek-r1-20250120",
    )
    result = asyncio.run(iface.get(ApproxOracleItem(
        key=0,
        query="""How would I use `asyncio.gather` with both a `concurrent.futures.ThreadPoolExecutor` and an asyncio loop? Please provide a full toy example that involves making a POST request (using `urllib.request`) to "http://api.example.com/".""",
    )))
    print(result)
    print(f"DEBUG: _approx_oracle: think:")
    print(result.think)
    print(f"DEBUG: _approx_oracle: value:")
    print(result.value)

def test_main_async_2():
    print(f"DEBUG: _approx_oracle: test main (async 2)")
    iface = ApproxOracleAsyncInterface(
        #default_model="deepseek-r1-20250120",
    )
    w1 = iface.get(ApproxOracleItem(
        key=0,
        query="""How would I use `asyncio.gather` with both a `concurrent.futures.ThreadPoolExecutor` and an asyncio loop? Please provide a full toy example that involves making a POST request (using `urllib.request`) to "http://api.example.com/".""",
    ))
    w2 = iface.get(ApproxOracleItem(
        key=0,
        query="""How would I implement a very basic parser for Test Anything Protocol (TAP)?""",
    ))
    async def _gather():
        return await asyncio.gather(w1, w2)
    results = asyncio.run(_gather())
    for r in results:
        print(r)

if __name__ == "__main__":
    #test_main()
    #test_main_async()
    test_main_async_2()
