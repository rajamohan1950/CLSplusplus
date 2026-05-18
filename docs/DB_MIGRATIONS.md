# Database Migrations

## How the schema is managed today

CLS++ has **no migration framework** (no Alembic/Flyway). The schema is
bootstrapped from DDL files that ship with the code:

- `src/clsplusplus/stores/*_ddl.sql` — one file per store (users, rbac,
  waitlist, prompt log, chat sessions, integrations, web events, metrics,
  metering v2).
- `scripts/init-pgvector.sql` — one-time `CREATE EXTENSION IF NOT EXISTS vector`.

Each store class applies its DDL on application startup. Every statement is
written as `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS`, so
startup is **idempotent for new tables** — a fresh database converges to the
correct schema, and an already-provisioned database is left untouched.

## What this does NOT handle

`CREATE TABLE IF NOT EXISTS` never alters an existing table. So **column
additions, type changes, renames, and drops are not applied automatically**.
If you edit a `*_ddl.sql` file to add a column, existing deployments will not
pick it up — the table already exists, so the `CREATE` is skipped.

There is also **no rollback path**: a bad schema change must be reverted by
hand.

## Procedure for an ALTER-type change

Until a real migration tool is adopted, apply schema changes explicitly:

1. **Write a numbered migration file.** Create
   `scripts/migrations/NNN_short_description.sql` (e.g.
   `001_add_users_last_seen.sql`), numbered sequentially. Use idempotent SQL
   where possible (`ADD COLUMN IF NOT EXISTS`, `DROP COLUMN IF EXISTS`).

2. **Include a rollback block** as a comment at the bottom of the file, so the
   reverse operation is recorded next to the forward one.

3. **Apply it to the target database** via `psql`:

   ```sh
   psql "$CLS_DATABASE_URL" -f scripts/migrations/001_add_users_last_seen.sql
   ```

   Run against staging first, then production. On Render, use the database's
   external connection string from the dashboard.

4. **Record that it ran.** Note the migration number and date in the PR
   description / deploy log. (There is no `schema_migrations` table yet — track
   applied migrations in the deploy notes.)

5. **Update the matching `*_ddl.sql` file** so a brand-new database created from
   scratch also gets the change. The `*_ddl.sql` files remain the source of
   truth for fresh installs; the `scripts/migrations/` files patch existing
   ones.

## When to adopt a real migration tool

If schema churn becomes frequent, adopt Alembic: it adds a `schema_migrations`
version table, automatic up/down, and removes the dual-maintenance of
`*_ddl.sql` + `scripts/migrations/`. That is deliberately out of scope for now
— this document makes the current manual process explicit and safe.
