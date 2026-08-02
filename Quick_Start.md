# 快速开始

## 1. 数据库准备

### 方式一：Docker Compose（推荐）

```bash
docker compose up -d
```

启动 PostgreSQL 16，创建用户 `psql_reading`、密码 `psql_reading`、数据库 `reading_agent`，与 `.env` 中的默认 `DATABASE_URL` 对应。

### 方式二：手动创建

```sql
-- 连接默认数据库
psql -d postgres -h localhost
-- 创建用户名密码、赋予角色
create user psql_reading with password '<your password>';
alter user psql_reading with superuser;
-- 创建数据库
CREATE DATABASE reading_agent OWNER psql_reading;
```

## 2. 建表

表由应用启动时自动创建（SQLAlchemy `create_all`，P2 阶段实现）。
