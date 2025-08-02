# 测试数据平台

针对各种测试数据进行存储和处理，展示测试过程和测试结果

## 环境启动

安装docker和docker-compose，启动开发环境

```bash
cd docker
docker-compose up -d
```

## 各中间件地址

| **中间件**       | **访问地址**                     | **用户/密码**          | **备注**        |
|---------------|------------------------------|--------------------|---------------|
| Kafka         | localhost:9094               |                    |               |
| Kafka-UI      | http://localhost:8080/       |                    | Kafka Web管理端  |
| MongoDB       | localhost:27017              | admin/admin        |               |
| Mongo Express | http://localhost:8081/       |                    | Mongo Web管理端  |
| InfluxDB      | http://localhost:8086/       |                    |               |
| Chronograf    | http://localhost:8888/       |                    | InfluxDB可视化插件 |
| Grafana       | http://localhost:3000/       | admin/admin        |               |
| Apache NiFi   | https://localhost:8443/nifi/ | admin/a12345678910 | 数据ETL工具       |

## Python SDK

[Python SDK 使用说明](./python-sdk/README.md)
