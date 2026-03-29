"""
Semantic tool filtering for the spatial reasoning agent.

This module provides semantic filtering of tools based on query relevance
using sentence transformers and tool retrieval systems.
"""

import logging
from typing import List, Dict, Any
from spatialreason.plan.tools import Tool

# Setup logger
logger = logging.getLogger("semantic_filter")

# Import retrieval dependencies with fallback
try:
    from spatialreason.plan.retriever import ToolRetriever
    from spatialreason.plan.utils import standardize, change_name
    RETRIEVAL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Retrieval dependencies not available: {e}")
    RETRIEVAL_AVAILABLE = False
    
    # Fallback implementations
    def standardize(text: str) -> str:
        return text.lower().replace(" ", "_")
    
    def change_name(text: str) -> str:
        return text.replace("_", "")


class SemanticToolFilter:
    """
    Semantic tool filter that integrates ToolRetriever with the Tool system.
    Provides semantic filtering of tools based on query relevance.
    """

    def __init__(self, corpus_tsv_path: str = "", model_path: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize semantic tool filter.

        Args:
            corpus_tsv_path: Path to tool corpus TSV file
            model_path: Path to sentence transformer model
        """
        # Store initialization parameters for refresh functionality
        self.corpus_tsv_path = corpus_tsv_path
        self.model_path = model_path
        self.retrieval_available = RETRIEVAL_AVAILABLE
        self.retriever = None

        if self.retrieval_available and corpus_tsv_path:
            try:
                logger.info("Initializing semantic tool retriever...")
                self.retriever = ToolRetriever(corpus_tsv_path, model_path)
                logger.info("Semantic tool retriever initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize semantic retriever: {e}")
                self.retrieval_available = False
        else:
            logger.info("Using simple tool filtering (no semantic retrieval)")

    def _detect_query_modality(self, query: str) -> str:
        """
        Detect the modality (SAR, IR, or optical) from query keywords.

        Args:
            query: The input query string

        Returns:
            'sar' - if query mentions SAR/radar
            'ir' - if query mentions IR/infrared/thermal
            'optical' - default for optical imagery
        """
        query_lower = query.lower()

        # Check for SAR indicators
        sar_keywords = ['sar', 'radar', 'synthetic aperture', 'backscatter', 'polarization', 'speckle']
        if any(kw in query_lower for kw in sar_keywords):
            return 'sar'

        # Check for IR indicators
        ir_keywords = ['ir', 'infrared', 'thermal', 'flir', 'mwir', 'lwir', 'heat signature', 'hotspot', 'small target']
        if any(kw in query_lower for kw in ir_keywords):
            return 'ir'

        # Default to optical
        return 'optical'

    def filter_tools_by_relevance(self, query: str, available_tools: List[Tool], top_k: int = 3, modality: str = None) -> List[Tool]:
        """
        Filter tools based on semantic relevance to the query.

        Args:
            query: The planning step query/description
            available_tools: List of available Tool objects
            top_k: Number of most relevant tools to return
            modality: Optional pre-detected modality ('sar', 'ir', or 'optical'). If not provided, will be detected from query.

        Returns:
            List of filtered Tool objects, ranked by relevance
        """
        if not self.retrieval_available or not self.retriever:
            # RESEARCH INTEGRITY: No fallback mechanisms - semantic filtering is required
            logger.error(f"❌ Semantic filtering unavailable but required for research integrity")
            logger.error(f"❌ Cannot proceed without semantic similarity-based tool selection")
            raise RuntimeError("Semantic filtering is required for research validity. No fallback mechanisms allowed.")

        try:
            # CRITICAL: Detect modality from query if not provided
            if modality is None:
                modality = self._detect_query_modality(query)
            else:
                logger.info(f"🔍 Using pre-detected modality: {modality.upper()}")
            logger.info(f"🔍 Query modality detected: {modality.upper()}")

            # Use semantic retrieval to get relevant tool names
            logger.info(f"Filtering {len(available_tools)} tools for query: '{query[:50]}...'")
            retrieved_tools = self.retriever.retrieving(query, top_k=top_k * 3)  # Get more candidates for modality filtering

            # Normalize all tool names to canonical form for consistent mapping
            tool_name_map = {}
            canonical_to_tool = {}

            for tool in available_tools:
                # Create canonical name using standardize + change_name for consistency
                canonical_name = change_name(standardize(tool.api_dest["name"]))
                canonical_package = change_name(standardize(tool.api_dest["package_name"]))

                # Use only canonical names for mapping
                tool_name_map[canonical_name] = tool
                tool_name_map[canonical_package] = tool
                canonical_to_tool[canonical_name] = tool

            # CRITICAL FIX: Prioritize modality-specific tools
            # Separate retrieved tools by modality
            modality_specific_tools = []
            other_tools = []

            for retrieved_tool in retrieved_tools:
                category = retrieved_tool.get('category', '')

                if modality == 'sar' and category == 'sar_tools':
                    modality_specific_tools.append(retrieved_tool)
                elif modality == 'ir' and category == 'ir_tools':
                    modality_specific_tools.append(retrieved_tool)
                elif modality == 'optical' and category not in ['sar_tools', 'ir_tools']:
                    modality_specific_tools.append(retrieved_tool)
                else:
                    other_tools.append(retrieved_tool)

            # Prioritize modality-specific tools, then add others
            prioritized_tools = modality_specific_tools + other_tools
            logger.info(f"🔍 Prioritized tools: {len(modality_specific_tools)} modality-specific + {len(other_tools)} others")

            # Match retrieved tools using canonical naming
            filtered_tools = []
            seen_canonical_names = set()

            logger.info(f"🔍 Retrieved {len(prioritized_tools)} tools from semantic search (prioritized by modality)")
            for idx, retrieved_tool in enumerate(prioritized_tools):
                logger.info(f"  [{idx+1}] {retrieved_tool['tool_name']} (category: {retrieved_tool['category']})")

                # Normalize retrieved tool names to canonical form
                canonical_candidates = [
                    change_name(standardize(retrieved_tool["tool_name"])),
                    change_name(standardize(retrieved_tool["api_name"]))
                ]

                for canonical_candidate in canonical_candidates:
                    if canonical_candidate in tool_name_map and canonical_candidate not in seen_canonical_names:
                        filtered_tools.append(tool_name_map[canonical_candidate])
                        seen_canonical_names.add(canonical_candidate)
                        logger.info(f"  ✅ Matched: {retrieved_tool['tool_name']} → {tool_name_map[canonical_candidate].api_dest['name']}")
                        break

                if len(filtered_tools) >= top_k:
                    logger.info(f"✅ Reached top_k={top_k}, stopping semantic filter")
                    break

            # If we didn't find enough matches, add remaining tools using canonical names
            if len(filtered_tools) < top_k:
                for tool in available_tools:
                    canonical_name = change_name(standardize(tool.api_dest["name"]))
                    if canonical_name not in seen_canonical_names:
                        filtered_tools.append(tool)
                        seen_canonical_names.add(canonical_name)
                        if len(filtered_tools) >= top_k:
                            break

            logger.info(f"Filtered to {len(filtered_tools)} most relevant tools")
            return filtered_tools

        except Exception as e:
            logger.error(f"❌ Semantic filtering failed: {e}")
            logger.error(f"❌ Research integrity requires semantic-only tool selection")
            raise RuntimeError(f"Semantic filtering failure compromises research validity: {e}")



    def refresh_retriever(self):
        """Refresh the semantic retriever with current parameters."""
        if self.retrieval_available and self.corpus_tsv_path:
            try:
                logger.info("Refreshing semantic tool retriever...")
                self.retriever = ToolRetriever(self.corpus_tsv_path, self.model_path)
                logger.info("Semantic tool retriever refreshed successfully")
            except Exception as e:
                logger.warning(f"Failed to refresh semantic retriever: {e}")
                self.retrieval_available = False

    def is_available(self) -> bool:
        """Check if semantic filtering is available."""
        return self.retrieval_available and self.retriever is not None

    def get_retriever_info(self) -> Dict[str, Any]:
        """Get information about the current retriever configuration."""
        return {
            "available": self.retrieval_available,
            "corpus_path": self.corpus_tsv_path,
            "model_path": self.model_path,
            "retriever_initialized": self.retriever is not None
        }
