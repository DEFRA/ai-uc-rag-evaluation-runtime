from logging import getLogger
from typing import Any

from aws_embedded_metrics import metric_scope
from aws_embedded_metrics.storage_resolution import StorageResolution

logger = getLogger(__name__)


# This is using the aws_embedded_metrics library, which doesn't seem to be playing nicely with fastapi
# metrics.put_metric always seems to thrown an exception, even though the metrics are being sent to cloudwatch
# This is a related issue: https://github.com/awslabs/aws-embedded-metrics-python/issues/52
# More time needs to be spent on this, but for now, the metrics are being sent to cloudwatch
# Setting metrics to Any as have no idea what this should be.
@metric_scope
def __put_metric(metric_name: str, value: float, unit: str, metrics: Any) -> None:
    logger.debug("put metric: %s - %s - %s", metric_name, value, unit)
    metrics.put_metric(metric_name, value, unit, StorageResolution.STANDARD)


# Use this counter function in the app, not the decorated function __put_metric.
# This wraps __put_metric and handles the exceptions, allows the app to continue running
#
# This does not set metric on the call to __put_metric not sure why?
def counter(metric_name: str, value: float) -> None:
    try:
        __put_metric(metric_name, value, "Count")
    except Exception as e:
        logger.error("Error calling put_metric: %s", e)
