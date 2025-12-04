from kafka import KafkaProducer, KafkaConsumer
import json

TOPIC = "agent-messages"
BOOTSTRAP = "localhost:9092"

def get_producer():
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

def get_consumer(group_id="coordinator-group"):
    return KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP,
        group_id=group_id,
        auto_offset_reset='earliest',
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        enable_auto_commit=True,
        consumer_timeout_ms=100  # <- MOST IMPORTANT LINE
    )