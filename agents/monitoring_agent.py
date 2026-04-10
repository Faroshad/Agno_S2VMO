#!/usr/bin/env python3
"""
Proactive structural monitoring agent.

Scans latest voxel state after each simulation cycle and writes alert artifacts
to Neo4j so the chatbot can surface actionable safety information.
"""

from typing import Dict, Any

from neo4j import GraphDatabase

# Support both package-relative and absolute imports
try:
    from ..config.settings import Settings
except ImportError:
    from config.settings import Settings


class MonitoringAgent:
    """Creates structured structural alerts from latest Neo4j voxel state."""

    def __init__(self) -> None:
        conn_info = Settings.get_connection_info()
        self.driver = GraphDatabase.driver(
            conn_info["uri"],
            auth=(conn_info["user"], conn_info["password"])
        )
        self.database = conn_info["database"]
        self.thresholds = Settings.get_alert_thresholds()
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Initialize alert schema once for deterministic alert writes."""
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                CREATE CONSTRAINT structural_alert_id IF NOT EXISTS
                FOR (a:StructuralAlert) REQUIRE a.alert_id IS UNIQUE
                """
            )
            session.run(
                """
                CREATE INDEX structural_alert_timestamp IF NOT EXISTS
                FOR (a:StructuralAlert) ON (a.timestamp)
                """
            )

    def assess_cycle(self, cycle: int, timestamp: str, limit: int = 10) -> Dict[str, Any]:
        """
        Evaluate latest structural state and create alerts for significant findings.

        Args:
            cycle: Simulation cycle number.
            timestamp: ISO timestamp for this cycle.
            limit: Maximum number of per-voxel alerts to emit.

        Returns:
            Dictionary with alert summary.
        """
        with self.driver.session(database=self.database) as session:
            stress_rows = session.run(
                """
                MATCH (v:Voxel)
                WHERE v.stress_magnitude IS NOT NULL AND size(v.stress_magnitude) > 0
                WITH v, v.stress_magnitude[size(v.stress_magnitude)-1] as current_stress
                WHERE current_stress >= $warning_threshold
                RETURN v.id as voxel_id,
                       v.grid_i as grid_i,
                       v.grid_j as grid_j,
                       v.grid_k as grid_k,
                       v.x as x,
                       v.y as y,
                       v.z as z,
                       v.type as voxel_type,
                       current_stress as stress_pa
                ORDER BY stress_pa DESC
                LIMIT $limit
                """,
                {
                    "warning_threshold": self.thresholds["stress_warning_pa"],
                    "limit": limit,
                },
            )
            stress_breaches = [dict(row) for row in stress_rows]

            quality_rows = session.run(
                """
                MATCH (v:Voxel)
                WHERE v.quality_ok = false
                RETURN v.id as voxel_id,
                       v.grid_i as grid_i,
                       v.grid_j as grid_j,
                       v.grid_k as grid_k,
                       v.quality_flags as quality_flags,
                       v.last_updated as last_updated
                ORDER BY v.last_updated DESC
                LIMIT $limit
                """,
                {"limit": limit},
            )
            quality_issues = [dict(row) for row in quality_rows]

            created_count = 0

            for row in stress_breaches:
                severity = (
                    "critical"
                    if row["stress_pa"] >= self.thresholds["stress_critical_pa"]
                    else "warning"
                )
                alert_id = f"cycle-{cycle}-stress-{row['voxel_id']}"

                session.run(
                    """
                    MERGE (a:StructuralAlert {alert_id: $alert_id})
                    ON CREATE SET a.created_at = $timestamp
                    SET a.timestamp = $timestamp,
                        a.cycle = $cycle,
                        a.alert_type = 'stress_threshold',
                        a.severity = $severity,
                        a.voxel_id = $voxel_id,
                        a.grid_i = $grid_i,
                        a.grid_j = $grid_j,
                        a.grid_k = $grid_k,
                        a.x = $x,
                        a.y = $y,
                        a.z = $z,
                        a.voxel_type = $voxel_type,
                        a.value = $stress_pa,
                        a.threshold_warning = $threshold_warning,
                        a.threshold_critical = $threshold_critical,
                        a.message = $message
                    """,
                    {
                        "alert_id": alert_id,
                        "timestamp": timestamp,
                        "cycle": cycle,
                        "severity": severity,
                        "voxel_id": row["voxel_id"],
                        "grid_i": row["grid_i"],
                        "grid_j": row["grid_j"],
                        "grid_k": row["grid_k"],
                        "x": row["x"],
                        "y": row["y"],
                        "z": row["z"],
                        "voxel_type": row["voxel_type"],
                        "stress_pa": row["stress_pa"],
                        "threshold_warning": self.thresholds["stress_warning_pa"],
                        "threshold_critical": self.thresholds["stress_critical_pa"],
                        "message": (
                            f"Voxel {row['voxel_id']} stress {row['stress_pa']:.2f} Pa "
                            f"exceeds {severity} threshold"
                        ),
                    },
                )
                created_count += 1

            for row in quality_issues:
                alert_id = f"cycle-{cycle}-quality-{row['voxel_id']}"
                session.run(
                    """
                    MERGE (a:StructuralAlert {alert_id: $alert_id})
                    ON CREATE SET a.created_at = $timestamp
                    SET a.timestamp = $timestamp,
                        a.cycle = $cycle,
                        a.alert_type = 'sensor_quality',
                        a.severity = 'warning',
                        a.voxel_id = $voxel_id,
                        a.grid_i = $grid_i,
                        a.grid_j = $grid_j,
                        a.grid_k = $grid_k,
                        a.quality_flags = $quality_flags,
                        a.last_updated = $last_updated,
                        a.message = $message
                    """,
                    {
                        "alert_id": alert_id,
                        "timestamp": timestamp,
                        "cycle": cycle,
                        "voxel_id": row["voxel_id"],
                        "grid_i": row["grid_i"],
                        "grid_j": row["grid_j"],
                        "grid_k": row["grid_k"],
                        "quality_flags": row.get("quality_flags"),
                        "last_updated": row.get("last_updated"),
                        "message": (
                            f"Voxel {row['voxel_id']} has sensor quality issues: "
                            f"{row.get('quality_flags', 'unknown')}"
                        ),
                    },
                )
                created_count += 1

            summary_alert_id = f"cycle-{cycle}-summary"
            max_stress = max((row["stress_pa"] for row in stress_breaches), default=0.0)
            session.run(
                """
                MERGE (a:StructuralAlert {alert_id: $alert_id})
                ON CREATE SET a.created_at = $timestamp
                SET a.timestamp = $timestamp,
                    a.cycle = $cycle,
                    a.alert_type = 'cycle_summary',
                    a.severity = $severity,
                    a.stress_breach_count = $stress_breach_count,
                    a.quality_issue_count = $quality_issue_count,
                    a.max_stress_pa = $max_stress_pa,
                    a.message = $message
                """,
                {
                    "alert_id": summary_alert_id,
                    "timestamp": timestamp,
                    "cycle": cycle,
                    "severity": "critical" if max_stress >= self.thresholds["stress_critical_pa"] else "info",
                    "stress_breach_count": len(stress_breaches),
                    "quality_issue_count": len(quality_issues),
                    "max_stress_pa": max_stress,
                    "message": (
                        f"Cycle {cycle}: {len(stress_breaches)} stress breaches, "
                        f"{len(quality_issues)} quality issues, max stress {max_stress:.2f} Pa"
                    ),
                },
            )

        return {
            "cycle": cycle,
            "timestamp": timestamp,
            "stress_breaches": len(stress_breaches),
            "quality_issues": len(quality_issues),
            "alerts_created": created_count + 1,
            "max_stress_pa": max((row["stress_pa"] for row in stress_breaches), default=0.0),
        }

    def close(self) -> None:
        self.driver.close()
