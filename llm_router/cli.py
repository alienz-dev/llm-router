import asyncio
import click
import uvicorn

from .config import get_config


@click.group()
def main():
    """LLM Router — Free-Tier LLM Gateway"""
    from .logging_setup import configure_logging

    configure_logging()


@main.command()
@click.option("--host", default=None, help="Host to bind to")
@click.option("--port", default=None, type=int, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host, port, reload):
    """Start the LLM Router server."""
    config = get_config()
    uvicorn.run(
        "llm_router.app:app",
        host=host or config.server.host,
        port=port or config.server.port,
        reload=reload,
    )


@main.command()
def discover():
    """Run model discovery once."""
    async def _run():
        from .db import get_db, close_db
        from .discovery import ModelDiscovery

        await get_db()
        # Seed providers
        from .app import _seed_providers
        await _seed_providers()

        d = ModelDiscovery()
        discovered = await d.discover_all()
        stats = await d.update_models_table(discovered)
        for provider, s in stats.items():
            click.echo(f"  {provider}: {s['active']} active (+{s['added']}, -{s['removed']})")
        await close_db()

    asyncio.run(_run())


@main.command()
@click.option("--seed-only", is_flag=True,
              help="Only apply the verified inventory file; make no provider calls")
@click.option("--overwrite", is_flag=True,
              help="Re-seed rows that already carry a capability reading")
def capabilities(seed_only, overwrite):
    """Seed and refresh per-model tool/schema capability flags."""
    async def _run():
        from .db import get_db, close_db
        from .app import _seed_providers
        from .capabilities import run_capability_probe, seed_from_inventory

        await get_db()
        await _seed_providers()
        seeded = await seed_from_inventory(overwrite=overwrite)
        click.echo(f"Seeded {seeded} models from the verified inventory")

        if not seed_only:
            stats = await run_capability_probe()
            click.echo(f"Probed {stats['probed']} models, updated {stats['updated']}")
            for r in stats["results"]:
                click.echo(f"  {r['provider']:12} {r['model'][:44]:44} "
                           f"tools={r['tools']:14} json={r['json']}")
        await close_db()

    asyncio.run(_run())


@main.command()
def status():
    """Show quota status for all providers."""
    async def _run():
        from .db import get_db, close_db
        from .app import _seed_providers
        from .quota import QuotaManager

        await get_db()
        await _seed_providers()
        qm = QuotaManager()
        statuses = await qm.get_all_quota_status()
        click.echo(f"{'Provider':<15} {'RPM':>10} {'TPM':>12} {'RPD':>10} {'Healthy':>8}")
        click.echo("-" * 58)
        for s in statuses:
            rpm = f"{s['rpm_used']}/{s['rpm_limit'] or '∞'}"
            tpm = f"{s['tpm_used']}/{s['tpm_limit'] or '∞'}"
            rpd = f"{s['rpd_used']}/{s['rpd_limit'] or '∞'}"
            healthy = "✅" if s["healthy"] else "❌"
            click.echo(f"{s['provider_id']:<15} {rpm:>10} {tpm:>12} {rpd:>10} {healthy:>8}")
        await close_db()

    asyncio.run(_run())


@main.command()
def process():
    """Process pending batch jobs once."""
    async def _run():
        from .db import get_db, close_db
        from .app import _build_adapters, _seed_providers
        from .quota import QuotaManager
        from .router import SmartRouter
        from .queue import JobQueue

        await get_db()
        await _seed_providers()
        adapters = _build_adapters()
        qm = QuotaManager()
        router = SmartRouter(adapters, qm)
        queue = JobQueue(router)
        n = await queue.process_batch_jobs()
        click.echo(f"Processed {n} batch jobs")
        await close_db()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
