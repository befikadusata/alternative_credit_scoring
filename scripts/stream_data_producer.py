import pandas as pd
from kafka import KafkaProducer
import json
import time
import argparse

def create_producer(bootstrap_servers):
    """Creates a Kafka producer."""
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        return producer
    except Exception as e:
        print(f"Error creating Kafka producer: {e}")
        return None

def stream_data(producer, topic, csv_file, rate=1):
    """Reads data from a CSV file and sends it to a Kafka topic."""
    df = pd.read_csv(csv_file)
    while True:
        for _, row in df.iterrows():
            message = row.to_dict()
            producer.send(topic, value=message)
            print(f"Sent message: {message}")
            time.sleep(rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafka data producer")
    parser.add_argument("--bootstrap-servers", default="kafka:9092", help="Kafka bootstrap servers")
    parser.add_argument("--topic", default="prediction_requests", help="Kafka topic to send messages to")
    parser.add_argument("--csv-file", default="data/raw/lending_club_loan.csv", help="Path to the CSV file")
    parser.add_argument("--rate", type=float, default=1, help="Rate in seconds to send messages")
    args = parser.parse_args()

    producer = create_producer(args.bootstrap_servers)
    if producer:
        stream_data(producer, args.topic, args.csv_file, args.rate)
