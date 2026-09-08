-- Runs once on first container start. Creates the test database and enables extensions
-- in both databases. Alembic also issues CREATE EXTENSION IF NOT EXISTS, so this is a
-- convenience for local development.
CREATE DATABASE outlier_test OWNER outlier;

\connect outlier
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

\connect outlier_test
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
