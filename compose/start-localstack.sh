#!/bin/bash
export AWS_REGION=eu-west-2
export AWS_DEFAULT_REGION=eu-west-2
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test

# SQS queues
aws --endpoint-url=http://localhost:4566 sqs create-queue \
  --queue-name rag_evaluation_start.fifo \
  --attributes FifoQueue=true,ContentBasedDeduplication=true


