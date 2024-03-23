import json
import aiohttp
import asyncio

from typing import Any, Dict, List, Optional
from uuid import UUID


class AgentClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.default_headers = {'Content-Type': 'application/json'}
        self.session = aiohttp.ClientSession()

    async def get(self, endpoint, params=None):
        async with self.session.get(f"{self.base_url}/{endpoint}", params=params, headers=self.default_headers) as response:
            response.raise_for_status()
            return await response.json()

    async def post(self, endpoint, data):
        async with self.session.post(f"{self.base_url}/{endpoint}", json=data, headers=self.default_headers) as response:
            response.raise_for_status()
            return await response.text()

    async def close(self):
        await self.session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

async def send_chat_query(client: AgentClient, chat_query: Dict) -> Dict[str, Any]:
    return await client.post("chat/agent", chat_query)



async def main():
    client = AgentClient("http://agentkit.web.local:8080/api/v1")
    chat_query = chat_query = {
            "messages": [
                {
                "role": "system",
                "content": "you are a helpful assistant"
                },
                {
                "role": "user",
                "content": "make a little poem about the sea"
                },
            ],
            "conversationId": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "newMessageId": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "userEmail": "string",
        }
    res = await send_chat_query(client, chat_query)
    print(res)
    await client.close()


    rs = res.split("\n")
    rsj = [json.loads(r) for r in rs]


def make_tree(rsj:List[Dict[str, Any]]):
    """
    every elements in rsj is a dict with keys: data, data_typ, metadata.parent_run_id,  and metadata.run_id

    root of the tree is the element with parent_run_id = None
    the tree is a dict with keys: metadata.run_id and children are the elements with parent_run_id = metadata.run_id

    
    """
    tree = {}
    nodes = {item['metadata']['run_id']: item for item in rsj}
    for run_id, node in nodes.items():
        parent_run_id = node['metadata']['parent_run_id']
        if parent_run_id is None:
            tree[run_id] = node
            tree[run_id]['children'] = []
        else:
            if 'children' not in nodes[parent_run_id]:
                nodes[parent_run_id]['children'] = []
            nodes[parent_run_id]['children'].append(node)
    return tree
        
if __name__ == "__main__":

    asyncio.run(main())
