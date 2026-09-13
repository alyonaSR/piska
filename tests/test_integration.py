"""
Интеграционные тесты полного цикла.

Зона ответственности: Person 1.
Проверяют ровно те три сценария, которые ТЗ требует показать на демо.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.contracts import REFUSE
from src.data.state_builder import build_demo_state
from src.orchestrator import Orchestrator


def test_normal_mode_produces_no_action():
    """Устойчивый режим: система не создаёт лишних управляющих действий."""
    t = Orchestrator().run_cycle(build_demo_state("normal"))
    rec = t.recommendation
    assert not rec.is_refusal
    assert all(abs(v) < 1e-9 for v in rec.action.values())


def test_quality_risk_produces_corrective_action():
    """Риск качества: система предлагает действие и оно проходит Gate."""
    t = Orchestrator().run_cycle(build_demo_state("quality_risk"))
    rec = t.recommendation
    assert not rec.is_refusal
    assert any(abs(v) > 1e-9 for v in rec.action.values())
    assert rec.expected_effect["sulfur_mgkg"]["hi"] <= 10.0


def test_degraded_data_produces_refusal():
    """Устаревшие и мёртвые данные: корректный отказ, а не риск."""
    t = Orchestrator().run_cycle(build_demo_state("degraded_data"))
    assert t.recommendation.action == REFUSE
    assert "устарел" in t.recommendation.reason or "неисправ" in t.recommendation.reason


def test_agents_actually_exchange_information():
    """Мультиагентность: allowed_ranges надёжности реально ограничивают оптимизатор."""
    t = Orchestrator().run_cycle(build_demo_state("normal"))
    assert t.reliability.allowed_ranges
    for c in t.candidates:
        for tag, d in c.deltas.items():
            lo, hi = t.reliability.allowed_ranges[tag]
            assert lo <= t.state.tag(tag) + d <= hi


def test_trace_is_serializable():
    """Воспроизводимость: весь цикл сериализуется в JSON."""
    import json
    t = Orchestrator().run_cycle(build_demo_state("quality_risk"))
    assert json.dumps(t.to_dict(), ensure_ascii=False)


def test_run_is_deterministic():
    a = Orchestrator().run_cycle(build_demo_state("quality_risk")).recommendation.action
    b = Orchestrator().run_cycle(build_demo_state("quality_risk")).recommendation.action
    assert a == b


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn(); print(f"  OK  {name}")
    print("\nвсе интеграционные тесты прошли")
