# 1. 数据库准备

## 创建角色和数据库
```sql
# 连接默认数据库
psql -d postgres -h localhost
# 创建用户名密码、赋予角色
create user psql_reading with password '<your password>';
alter user psql_reading with superuser;
# 创建数据库
CREATE DATABASE reading_agent OWNER psql_reading;
```

## 创建表
- 自动创建



