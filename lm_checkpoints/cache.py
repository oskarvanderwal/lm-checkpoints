"""Cache management for HuggingFace model checkpoints."""

import logging
from typing import Literal, Optional

from huggingface_hub import scan_cache_dir
from huggingface_hub.errors import CacheNotFound


logger = logging.getLogger(__name__)

CachePolicy = Literal["keep", "previous", "bounded"]


class CacheManager:
    """Manages HuggingFace cache for checkpoint iteration.

    Args:
        policy: Cache management policy:
            - "keep": Default HF behavior, keep all downloaded checkpoints.
            - "previous": Delete previous checkpoint after loading next one.
            - "bounded": Prune oldest cached models when cache exceeds max_size_gb.
        cache_dir: Custom cache directory. If None, uses default HF cache.
        max_size_gb: Maximum cache size in GB (required for policy="bounded").
    """

    def __init__(
        self,
        policy: CachePolicy = "keep",
        cache_dir: Optional[str] = None,
        max_size_gb: Optional[float] = None,
    ):
        if policy not in ("keep", "previous", "bounded"):
            raise ValueError(
                f"Invalid cache policy: {policy}. "
                "Must be one of: 'keep', 'previous', 'bounded'"
            )

        if policy == "bounded" and max_size_gb is None:
            raise ValueError("max_size_gb is required when policy='bounded'")

        self.policy = policy
        self.cache_dir = cache_dir
        self.max_size_gb = max_size_gb
        self._pending_deletions: list[str] = []

    @staticmethod
    def get_cache_size_gb(cache_dir: Optional[str] = None) -> float:
        """Get the current HuggingFace cache size in GB."""
        try:
            cache_info = scan_cache_dir(cache_dir=cache_dir)
            return cache_info.size_on_disk / (1024**3)
        except CacheNotFound:
            return 0.0

    def get_revision_hash(self, repo_id: str, revision: str) -> Optional[str]:
        """Get the commit hash for a repo/revision from the cache."""
        try:
            cache_info = scan_cache_dir(cache_dir=self.cache_dir)
            for repo in cache_info.repos:
                if repo.repo_id == repo_id and revision in repo.refs:
                    return repo.refs[revision].commit_hash
        except CacheNotFound:
            pass
        return None

    def enforce_limit(self) -> None:
        """Prune oldest cached revisions if cache exceeds max_size_gb."""
        if self.policy != "bounded" or self.max_size_gb is None:
            return

        try:
            cache_info = scan_cache_dir(cache_dir=self.cache_dir)
        except CacheNotFound:
            return

        current_size_gb = cache_info.size_on_disk / (1024**3)

        if current_size_gb <= self.max_size_gb:
            return

        logger.info(
            f"Cache size ({current_size_gb:.2f}GB) exceeds limit ({self.max_size_gb}GB), "
            "pruning oldest revisions"
        )

        # Collect all revisions with their last accessed time
        revisions = []
        for repo in cache_info.repos:
            for revision in repo.revisions:
                revisions.append((
                    revision.last_accessed,
                    revision.commit_hash,
                    revision.size_on_disk,
                    repo.repo_id,
                ))

        # Sort by last accessed time (oldest first)
        revisions.sort(key=lambda x: x[0])

        # Delete oldest revisions until under limit
        deleted_count = 0
        failed_count = 0
        for _, commit_hash, size, repo_id in revisions:
            if current_size_gb <= self.max_size_gb:
                break
            try:
                delete_strategy = cache_info.delete_revisions(commit_hash)
                delete_strategy.execute()
                current_size_gb -= size / (1024**3)
                deleted_count += 1
                logger.debug(f"Deleted cached revision {commit_hash[:8]} from {repo_id}")
            except Exception as e:
                failed_count += 1
                logger.warning(f"Failed to delete cache revision {commit_hash[:8]}: {e}")

        if deleted_count > 0:
            logger.info(f"Pruned {deleted_count} cached revision(s), cache now {current_size_gb:.2f}GB")
        if failed_count > 0:
            logger.warning(f"Failed to delete {failed_count} revision(s)")

    def delete_revision(self, commit_hash: str) -> bool:
        """Delete a specific revision from the cache."""
        try:
            cache_info = scan_cache_dir(cache_dir=self.cache_dir)
            delete_strategy = cache_info.delete_revisions(commit_hash)
            delete_strategy.execute()
            logger.debug(f"Deleted cached revision {commit_hash[:8]}")
            return True
        except CacheNotFound:
            logger.debug(f"Cache not found when trying to delete revision {commit_hash[:8]}")
            return False
        except Exception as e:
            logger.warning(f"Failed to delete cache revision {commit_hash[:8]}: {e}")
            return False

    def mark_for_deletion(self, commit_hash: str) -> None:
        """Mark a revision for deletion on next flush (for 'previous' policy)."""
        if commit_hash:
            self._pending_deletions.append(commit_hash)

    def flush_pending_deletions(self) -> None:
        """Delete all revisions marked for deletion."""
        for commit_hash in self._pending_deletions:
            self.delete_revision(commit_hash)
        self._pending_deletions.clear()

    def on_pre_load(self) -> None:
        """Called before loading a checkpoint."""
        if self.policy == "bounded":
            self.enforce_limit()
        elif self.policy == "previous":
            self.flush_pending_deletions()

    def on_post_load(self, commit_hash: Optional[str] = None) -> None:
        """Called after loading a checkpoint."""
        if self.policy == "bounded":
            self.enforce_limit()
        elif self.policy == "previous" and commit_hash:
            self.mark_for_deletion(commit_hash)
