from prometheus_client import Counter, Histogram, start_http_server
import time


REQUEST_COUNT = Counter ('api_requests_total', 'Total API requests')
LATENCY = Histogram ('api_latency_seconds', 'API latency')
JOB_STATUS = Counter ('job_status_total', 'Job status counts', ['status'])


class MonitoringMiddleware:
    async def __call__(self, request, call_next):
        REQUEST_COUNT.inc ()
        start_time = time.time ()
        response = await call_next (request)
        LATENCY.observe (time.time () - start_time)
        return response


def start_metrics_server():
    start_http_server (8000)
