# Elasticsearch

## api requests using curl

list all indices
```bash
curl -X GET "es:9200/_cat/indices?v"
```
health status index         uuid                   pri rep docs.count docs.deleted store.size pri.store.size
yellow open   f4r           OIvd-cqLQX62o90MKYCkdg   1   1          0            0       282b           282b
yellow open   f4r_user_info tUArcDqtQH-NcOPvV79UhQ   1   1         31            4     19.9kb         19.9kb

Count documents in index
```bash
curl -X GET "es:9200/f4r_user_info/_count"
curl -X GET "es:9200/f4r/_count?pretty"
```

{"count":31,"_shards":{"total":1,"successful":1,"skipped":0,"failed":0}}

List documents in index
```bash
curl -X GET "es:9200/f4r_user_info/_search"
```

List documents in index with limit
```bash
curl -X GET "es:9200/f4r_user_info/_search?size=50"
curl -X GET "es:9200/f4r/_search?size=50"
```

List documents in index with limit and pretty print
```bash
curl -X GET "es:9200/f4r_user_info/_search?size=50&pretty"
```

List documents in index with query
```bash
curl -X GET "es:9200/f4r_user_info/_search?q=user_id:customer_00032_1757538126023"
```

Get document by id
```bash
curl -X GET "es:9200/f4r_user_info/_doc/customer_00017_1757537249192"
```