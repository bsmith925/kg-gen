from kg_gen.models import Graph
from kg_gen.utils.deduplicate import run_semhash_deduplication
from kg_gen.utils.llm_deduplicate import LLMDeduplicate
from sentence_transformers import SentenceTransformer
import dspy
import enum


class DeduplicateMethod(enum.Enum):
    SEMHASH = "semhash"  # Deduplicate using deterministic rules and semantic hashing
    LM_BASED = (
        "lm_based"  # Deduplicate using KNN clustering + Intra cluster LM deduplication
    )
    FULL = "full"  # Deduplicate using both semantic hashing and KNN clustering + Intra cluster LM deduplication


def run_deduplication(
    lm: dspy.LM,
    graph: Graph,
    method: DeduplicateMethod = DeduplicateMethod.FULL,
    retrieval_model: SentenceTransformer | None = None,
    semhash_similarity_threshold: float = 0.95,
    timeout_per_cluster: float = 300.0,
    max_iterations_per_cluster: int = 10000,
) -> Graph:
    """
    Run deduplication on a graph.

    Args:
        lm: DSPy language model
        graph: Graph to deduplicate
        method: Deduplication method to use
        retrieval_model: Sentence transformer model for embeddings
        semhash_similarity_threshold: Threshold for semhash deduplication (0.0-1.0)
        timeout_per_cluster: Maximum time in seconds per cluster (default: 300s)
        max_iterations_per_cluster: Maximum iterations in deduplicate_cluster loop (default: 10000)

    Returns:
        Deduplicated graph
    """
    if method != DeduplicateMethod.SEMHASH and retrieval_model is None:
        raise ValueError("No retrieval model provided")

    if method == DeduplicateMethod.SEMHASH:
        deduplicated_graph = run_semhash_deduplication(
            graph, semhash_similarity_threshold
        )
    elif method == DeduplicateMethod.LM_BASED:
        llm_deduplicate = LLMDeduplicate(
            retrieval_model, lm, graph, max_iterations_per_cluster
        )
        llm_deduplicate.cluster()
        deduplicated_graph = llm_deduplicate.deduplicate(
            timeout_per_cluster=timeout_per_cluster
        )
    elif method == DeduplicateMethod.FULL:
        deduplicated_graph = run_semhash_deduplication(
            graph, semhash_similarity_threshold
        )
        llm_deduplicate = LLMDeduplicate(
            retrieval_model, lm, deduplicated_graph, max_iterations_per_cluster
        )
        llm_deduplicate.cluster()
        deduplicated_graph = llm_deduplicate.deduplicate(
            timeout_per_cluster=timeout_per_cluster
        )

    return deduplicated_graph
