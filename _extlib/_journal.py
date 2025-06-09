from typing import Any, Union
import asyncio
from dataclasses import dataclass, asdict
import json
import struct

@dataclass
class JournalAsyncInterface:
    sort: str = None

    _conn_rx: Any = None
    _conn_tx: Any = None

    def __post_init__(self):
        assert self.sort is not None, (
            "JournalAsyncInterface: expected non-None sort"
        )

    async def _lazy_init(self):
        if self._conn_tx is not None:
            return
        assert self._conn_rx is None
        self._conn_rx, self._conn_tx = await asyncio.open_connection("127.0.0.1", 9001)

    async def hi(self):
        await self._lazy_init()
        req_data = b"hi \n"
        self._conn_tx.write(req_data)
        res_data = await self._conn_rx.read(4)
        if res_data == b"ok \n":
            return True
        elif res_data == b"err\n":
            return False
        else:
            return None

    async def put(self, item: Union[Any, dict]):
        await self._lazy_init()
        if not isinstance(item, dict):
            item = asdict(item)
        data = json.dumps(item).encode("utf-8")
        req_data = b"put\n" + self.sort.encode("utf-8") + b"\n" + data
        self._conn_tx.write(req_data)
        res_data = await self._conn_rx.read(4)
        if res_data == b"ok \n":
            return "ok"
        elif res_data == b"err\n":
            return "err"
        else:
            return None

    async def get(self, item: Union[Any, dict]) -> tuple[str, Any]:
        print(f"DEBUG: JournalAsyncInterface.get: ...")
        await self._lazy_init()
        print(f"DEBUG: JournalAsyncInterface.get: lazy init: done")
        if not isinstance(item, dict):
            item = asdict(item)
        data = json.dumps(item).encode("utf-8")
        req_data = b"get\n" + self.sort.encode("utf-8") + b"\n" + data
        print(f"DEBUG: JournalAsyncInterface.get: write...")
        self._conn_tx.write(req_data)
        print(f"DEBUG: JournalAsyncInterface.get: read...")
        res_data = await self._conn_rx.readexactly(4)
        print(f"DEBUG: JournalAsyncInterface.get: read len = {len(res_data)}")
        if len(res_data) < 4:
            print(f"DEBUG: JournalAsyncInterface.get: eof: len = {len(res_data)}")
            return "eof", None
        elif res_data == b"ok \n":
            res_data = await self._conn_rx.readexactly(4)
            res_len = struct.unpack("<I", res_data)[0]
            res_item = None
            if res_len > 0:
                res_data = await self._conn_rx.readexactly(res_len)
                res_str = None
                try:
                    res_str = res_data.decode("utf-8")
                    res_item = json.loads(res_str)
                except Exception as e:
                    print(f"DEBUG: JournalAsyncInterface.get: except = {e} s = {repr(res_str)}")
                    res_item = None
                # res_item |= item
            return "ok", res_item
        elif res_data == b"err\n":
            return "err", None
        else:
            return "except", None

def _main():
    async def _run():
        iface = JournalAsyncInterface("test")
        print(f"DEBUG: _extlib._journal._main: hi...")
        ret = await iface.hi()
        print(f"DEBUG: _extlib._journal._main: hi: ret = {ret}")
        print(f"DEBUG: _extlib._journal._main: get...")
        ret = await iface.get({"hello": "world"})
        print(f"DEBUG: _extlib._journal._main: get: ret = {ret}")
    asyncio.run(_run())

if __name__ == "__main__":
    _main()
