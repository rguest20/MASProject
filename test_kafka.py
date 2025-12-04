from kafka import KafkaProducer

try:
    p = KafkaProducer(bootstrap_servers="localhost:9092")
    print("✅ Kafka connection successful.")
except Exception as e:
    print("❌ Kafka connection failed:", e)