"""Cache management for HuggingFace model checkpoints."""

import logging
from typing import Literal, Optional

from huggingface_hub import scan_cache_dir
from huggingface_hub.errors import CacheNotFound


logger = logging.getLogger(__name__)

CachePolicy = Literal["keep", "previous"]


class CacheManager:
    """Manages HuggingFace cache for checkpoint iteration.

    Args:
        policy: Cache management policy:
            - "keep": Default HF behavior, keep all downloaded checkpoints.
            - "previous": Delete previous checkpoint after loading next one.
        cache_dir: Custom cache directory. If None, uses default HF cache.
    """

    def __init__(
        self,
        policy: CachePolicy = "keep",
        cache_dir: Optional[str] = None,
    ):
        if policy not in ("keep", "previous"):
            raise ValueError(
                f"Invalid cache policy: {policy}. "
                "Must be one of: 'keep', 'previous'"
            )

        self.policy = policy
        self.cache_dir = cache_dir
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
        if self.policy == "previous":
            self.flush_pending_deletions()

    def on_post_load(self, commit_hash: Optional[str] = None) -> None:
        """Called after loading a checkpoint."""
        if self.policy == "previous" and commit_hash:
            self.mark_for_deletion(commit_hash)
