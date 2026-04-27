# setup_topics.py

from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

KAFKA_BROKER = "localhost:9092"

TOPICS = [
    "usgs-earthquakes",
    "nasa-eonet-events",
    "fema-disasters",
    "processed-disasters"
]

def create_topics():
    admin = KafkaAdminClient(bootstrap_servers=KAFKA_BROKER)
    topic_list = [
        NewTopic(name=t, num_partitions=3, replication_factor=1)
        for t in TOPICS
    ]
    for topic in topic_list:
        try:
            admin.create_topics([topic])
            print(f"Created topic: {topic.name}")
        except TopicAlreadyExistsError:
            print(f"Topic already exists: {topic.name}")
    admin.close()

def delete_topics():
    admin = KafkaAdminClient(bootstrap_servers=KAFKA_BROKER)
    admin.delete_topics(TOPICS)
    print("Deleted topics:", ", ".join(TOPICS))
    admin.close()

if __name__ == "__main__":
    # To delete topics
    # delete_topics()
    create_topics()