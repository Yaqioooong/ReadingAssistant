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

> 注意：`create_all` 只会创建**不存在的表**，不会给已存在的表加列或改类型。
> 若模型变更后启动报 `UndefinedColumn` 之类的错误，说明本地库中有旧版结构的表；
> 开发环境可先备份数据，再删除对应表让应用重建（空表直接删即可）。
