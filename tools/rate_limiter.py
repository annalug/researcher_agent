"""
Rate Limiter for API calls
Ensures compliance with API rate limits (e.g., Semantic Scholar: 1 req/sec)
"""
import time
from datetime import datetime
from threading import Lock


class RateLimiter:
    """
    Simple rate limiter that enforces minimum time between requests.
    Thread-safe implementation.
    """

    def __init__(self, min_interval: float = 1.0):
        """
        Initialize rate limiter.

        Args:
            min_interval: Minimum seconds between requests (default: 1.0 for Semantic Scholar)
        """
        self.min_interval = min_interval
        self.last_request_time = 0
        self.lock = Lock()
        self.request_count = 0

    def wait_if_needed(self):
        """
        Wait if necessary to respect rate limit.
        Call this BEFORE making an API request.
        """
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                print(f"⏳ Rate limit: waiting {wait_time:.2f}s...")
                time.sleep(wait_time)

            self.last_request_time = time.time()
            self.request_count += 1

    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "total_requests": self.request_count,
            "last_request": datetime.fromtimestamp(
                self.last_request_time).isoformat() if self.last_request_time > 0 else None,
            "rate_limit": f"1 request per {self.min_interval} seconds"
        }


# Global rate limiter for Semantic Scholar (1 req/sec)
semantic_scholar_limiter = RateLimiter(min_interval=1.0)

if __name__ == "__main__":
    # Test the rate limiter
    print("Testing rate limiter (1 req/sec)...\n")

    limiter = RateLimiter(min_interval=1.0)

    for i in range(5):
        print(f"Request {i + 1}:", end=" ")
        start = time.time()
        limiter.wait_if_needed()
        elapsed = time.time() - start
        print(f"executed after {elapsed:.2f}s")

    print("\nStats:", limiter.get_stats())