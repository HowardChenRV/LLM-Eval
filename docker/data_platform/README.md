## Data Platform (data_platform)

For storing and processing various test data, displaying test processes and results.

Environment startup:
```bash
cd docker/data_platform
docker-compose up -d
```

Middleware addresses:

| **Middleware**    | **Access Address**           | **Username/Password**  | **Remarks**              |
|-------------------|------------------------------|------------------------|--------------------------|
| Kafka             | localhost:9094               |                        |                          |
| Kafka-UI          | http://localhost:8080/       |                        | Kafka Web Admin Console  |
| MongoDB           | localhost:27017              | admin/admin            |                          |
| Mongo Express     | http://localhost:8081/       |                        | Mongo Web Admin Console  |
| InfluxDB          | http://localhost:8086/       |                        |                          |
| Chronograf        | http://localhost:8888/       |                        | InfluxDB Visualization Plugin |
| Grafana           | http://localhost:3000/       | admin/admin            |                          |
| Apache NiFi       | https://localhost:8443/nifi/ | admin/a12345678910     | Data ETL Tool            |
