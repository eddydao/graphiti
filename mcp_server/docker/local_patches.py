"""Build-time fixes applied to graphiti-core and the MCP server source in the local image.

Each replacement must match exactly once so an upstream change fails the build loudly
instead of silently shipping unpatched code.
"""

import pathlib
import sys

import graphiti_core

CORE = pathlib.Path(graphiti_core.__file__).parent
MCP_SRC = pathlib.Path('/app/mcp/src')

PATCHES = [
    (
        CORE / 'graphiti.py',
        """                if update_communities:
                    communities, community_edges = await semaphore_gather(
                        *[
                            update_community(
                                driver,
                                clients.llm_client,
                                clients.embedder,
                                node,
                                clients=clients,
                            )
                            for node in nodes
                        ],
                        max_coroutines=self.max_coroutines,
                    )
""",
        """                if update_communities:
                    community_results = await semaphore_gather(
                        *[
                            update_community(
                                driver,
                                clients.llm_client,
                                clients.embedder,
                                node,
                                clients=clients,
                            )
                            for node in nodes
                        ],
                        max_coroutines=self.max_coroutines,
                    )
                    communities = [c for cs, _ in community_results for c in cs]
                    community_edges = [e for _, es in community_results for e in es]
""",
    ),
    (
        CORE / 'utils/maintenance/community_operations.py',
        """    community_map = {uuid: i for i, uuid in enumerate(projection.keys())}

    while True:
""",
        """    community_map = {uuid: i for i, uuid in enumerate(projection.keys())}

    # Synchronous label propagation can oscillate forever on bipartite-like graphs.
    for _ in range(max(100, 2 * len(projection))):
""",
    ),
    (
        CORE / 'cross_encoder/openai_reranker_client.py',
        """            for top_logprobs in responses_top_logprobs:
                if len(top_logprobs) == 0:
                    continue
""",
        """            for top_logprobs in responses_top_logprobs:
                if len(top_logprobs) == 0:
                    scores.append(0.5)
                    continue
""",
    ),
    (
        MCP_SRC / 'services/factories.py',
        """                return OpenAIRerankerClient(
                    config=GraphitiLLMConfig(
                        api_key=config.providers.openai.api_key,
                        base_url=config.providers.openai.api_url,
                    )
                )
""",
        """                return OpenAIRerankerClient(
                    config=GraphitiLLMConfig(
                        api_key=config.providers.openai.api_key,
                        base_url=config.providers.openai.api_url,
                        model=os.environ.get('RERANKER_MODEL') or None,
                    )
                )
""",
    ),
    (
        MCP_SRC / 'services/factories.py',
        'import re\n',
        'import os\nimport re\n',
    ),
    (
        MCP_SRC / 'services/factories.py',
        """        # Try the LLM provider first, then the embedder, before falling back to a local model.
""",
        """        reranker_model = os.environ.get('RERANKER_MODEL', '')
        if reranker_model.startswith('typesafe/') and embedder_config.providers.openai:
            from graphiti_core.cross_encoder.jev_reranker_client import JevRerankerClient

            logger.info(f'Using JevRerankerClient ({reranker_model}) via embedder endpoint')
            return JevRerankerClient(
                api_key=embedder_config.providers.openai.api_key,
                model=reranker_model,
                base_url=embedder_config.providers.openai.api_url,
            )

        # Try the LLM provider first, then the embedder, before falling back to a local model.
""",
    ),
    (
        MCP_SRC / 'graphiti_mcp_server.py',
        """        node_config = (
            NODE_HYBRID_SEARCH_NODE_DISTANCE if center_node_uuid else NODE_HYBRID_SEARCH_RRF
        )
        results = await client.search_(
            query=query,
            config=node_config,
            group_ids=effective_group_ids,
            center_node_uuid=center_node_uuid,
            search_filter=search_filters,
        )
""",
        """        node_config = (
            NODE_HYBRID_SEARCH_NODE_DISTANCE if center_node_uuid else NODE_HYBRID_SEARCH_RRF
        )
        results = None
        if not center_node_uuid and os.environ.get('RERANK_SEARCH', '0') == '1':
            from graphiti_core.search.search_config_recipes import (
                NODE_HYBRID_SEARCH_CROSS_ENCODER,
            )

            try:
                results = await client.search_(
                    query=query,
                    config=NODE_HYBRID_SEARCH_CROSS_ENCODER.model_copy(
                        update={'limit': max_nodes}
                    ),
                    group_ids=effective_group_ids,
                    search_filter=search_filters,
                )
            except Exception as rerank_error:
                logger.warning(f'Reranked node search failed, using RRF: {rerank_error}')
        if results is None:
            results = await client.search_(
                query=query,
                config=node_config,
                group_ids=effective_group_ids,
                center_node_uuid=center_node_uuid,
                search_filter=search_filters,
            )
""",
    ),
    (
        MCP_SRC / 'graphiti_mcp_server.py',
        """        relevant_edges = await client.search(
            group_ids=effective_group_ids,
            query=query,
            num_results=max_facts,
            center_node_uuid=center_node_uuid,
            search_filter=search_filter,
        )
""",
        """        relevant_edges = None
        if not center_node_uuid and os.environ.get('RERANK_SEARCH', '0') == '1':
            from graphiti_core.search.search_config_recipes import (
                EDGE_HYBRID_SEARCH_CROSS_ENCODER,
            )

            try:
                reranked = await client.search_(
                    query=query,
                    config=EDGE_HYBRID_SEARCH_CROSS_ENCODER.model_copy(
                        update={'limit': max_facts}
                    ),
                    group_ids=effective_group_ids,
                    search_filter=search_filter,
                )
                relevant_edges = reranked.edges
            except Exception as rerank_error:
                logger.warning(f'Reranked fact search failed, using RRF: {rerank_error}')
        if relevant_edges is None:
            relevant_edges = await client.search(
                group_ids=effective_group_ids,
                query=query,
                num_results=max_facts,
                center_node_uuid=center_node_uuid,
                search_filter=search_filter,
            )
""",
    ),
]


def main() -> int:
    jev_src = pathlib.Path(__file__).with_name('jev_reranker_client.py')
    (CORE / 'cross_encoder/jev_reranker_client.py').write_text(jev_src.read_text())
    print('installed jev_reranker_client.py')
    for path, old, new in PATCHES:
        text = path.read_text()
        count = text.count(old)
        if count != 1:
            print(f'patch target matched {count} times in {path}', file=sys.stderr)
            return 1
        path.write_text(text.replace(old, new))
        print(f'patched {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
