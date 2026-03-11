# Session: Project Planning — 2026-03-08

**User:** I need to set up a CI/CD pipeline for our new microservice. It's a Python FastAPI app that talks to PostgreSQL. What's the best approach?

**Assistant:** For a FastAPI + PostgreSQL stack, I'd recommend GitHub Actions with three stages: lint/test, build container, deploy. For the database, use Alembic migrations triggered before deployment. Key decisions: do you want blue-green deployments or rolling updates?

**User:** Rolling updates are fine for now. We're small — 3 engineers. What about the database migrations? We've had issues where migrations break production.

**Assistant:** Two safeguards: First, run migrations in a pre-deploy step with a timeout — if the migration takes longer than 60 seconds, abort. Second, require that every migration has a corresponding downgrade. Alembic supports this natively. For your team size, I'd also add a Slack notification before any migration runs in production.

**User:** That makes sense. Can we also add staging? We want to test migrations there first.

**Assistant:** Yes — add a staging environment that mirrors production. Your pipeline becomes: PR → tests → merge to main → deploy staging → run migrations → smoke tests → manual approval → deploy production. The manual approval gate is important at your scale — it costs nothing and catches deployment mistakes.

**User:** Perfect. Let's go with that. One more thing — we need to handle secrets. API keys, database credentials.

**Assistant:** Use GitHub Actions secrets for CI/CD. For the running application, use environment variables injected by your deployment platform. Never put secrets in the container image. If you grow past 3 engineers, consider moving to HashiCorp Vault or AWS Secrets Manager, but that's premature for now.
