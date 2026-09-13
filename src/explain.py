"""
Сборка объяснения оператору.

Зона ответственности: Person 1.

ВАЖНО: текст собирается ИЗ ПОЛЕЙ структур, а не сочиняется.
Нельзя показать на демо красивое объяснение, не подтверждённое числами.

Если подключаете LLM — он получает на вход уже готовый Recommendation
и только переписывает его человеческим языком. Права менять числа
или добавлять новые утверждения у него нет.
"""

from __future__ import annotations

from typing import Any, Dict

from .contracts import ProcessState, Recommendation


def summarize_freshness(state: ProcessState) -> Dict[str, Any]:
    """Блок 'Время и состояние' из раздела 5 ТЗ."""
    out: Dict[str, Any] = {"ts": state.ts.isoformat(), "dq_flags": list(state.dq_flags)}
    for name, store in (("lims", state.lims), ("pak", state.pak)):
        for param, m in store.items():
            out[f"{name}:{param}"] = {
                "value": m.value,
                "age_min": m.age_min,
                "healthy": m.healthy,
                "units": m.units,
            }
    return out


def render_explanation(rec, quality, reliability, candidate, verdict, n_survivors: int) -> str:
    parts = []

    if candidate.is_no_action:
        parts.append("Рекомендация: режим не менять.")
    else:
        moves = ", ".join(
            f"{tag} на {d:+.1f}" for tag, d in candidate.deltas.items() if abs(d) > 1e-9
        )
        parts.append(f"Рекомендация: изменить {moves}.")

    parts.append(f"Причина: {rec.reason}.")

    if quality.drivers:
        parts.append("Факторы: " + "; ".join(quality.drivers[:3]) + ".")

    s = rec.expected_effect.get("sulfur_mgkg", {})
    margin = rec.expected_effect.get("margin_to_spec")
    if s:
        parts.append(
            f"Ожидаемая сера {s['mean']} мг/кг, верхняя граница прогноза {s['hi']}, "
            f"запас до лимита {margin} мг/кг."
        )

    parts.append(
        f"Тяжесть режима {reliability.severity_class} "
        f"(индекс {reliability.severity_index}), изменение {candidate.severity_delta:+.3f}."
    )
    parts.append(f"Проверено ограничений: {len(verdict.checked)}, нарушений нет.")
    parts.append(
        f"Допустимых альтернатив рассмотрено: {n_survivors}. "
        "Выбран вариант с наибольшим запасом по сере при наименьшей тяжести режима."
    )
    parts.append(f"Доверие к прогнозу: {rec.confidence:.2f}.")

    return " ".join(parts)


def print_operator_report(rec: Recommendation) -> str:
    """Печать отчёта в терминал. Структура 1:1 с таблицей раздела 5 ТЗ."""
    lines = []
    w = 78
    lines.append("=" * w)
    lines.append(f"РЕКОМЕНДАЦИЯ ОПЕРАТОРУ   {rec.ts:%Y-%m-%d %H:%M}")
    lines.append("=" * w)

    lines.append("\n[1] ВРЕМЯ И СОСТОЯНИЕ")
    for k, v in rec.data_freshness.items():
        if k in ("ts", "dq_flags"):
            continue
        age = v.get("age_min")
        age_s = f"{age/60:.1f} ч" if age is not None else "нет"
        hp = "" if v.get("healthy", True) else "  [НЕИСПРАВЕН]"
        lines.append(f"    {k:<24} {str(v.get('value')):>10} {v.get('units','')}"
                     f"   возраст {age_s}{hp}")
    flags = rec.data_freshness.get("dq_flags") or []
    for f in flags:
        lines.append(f"    ! {f}")

    lines.append("\n[2] ПРОБЛЕМА / РИСК")
    lines.append(f"    {rec.reason}")

    lines.append("\n[3] ПРЕДЛАГАЕМОЕ ДЕЙСТВИЕ")
    if rec.is_refusal:
        lines.append("    ОТКАЗ ОТ РЕКОМЕНДАЦИИ")
    elif not rec.action:
        lines.append("    изменений не требуется")
    else:
        for tag, d in rec.action.items():
            mark = "" if abs(d) > 1e-9 else "   (без изменений)"
            lines.append(f"    {tag:<20} {d:+.2f}{mark}")

    lines.append("\n[4] ОЖИДАЕМЫЙ ЭФФЕКТ")
    if not rec.expected_effect:
        lines.append("    не оценивается")
    for k, v in rec.expected_effect.items():
        lines.append(f"    {k:<20} {v}")

    lines.append("\n[5] ПРОВЕРКА ОГРАНИЧЕНИЙ")
    if not rec.checks_passed:
        lines.append("    проверки не выполнялись")
    for c in rec.checks_passed:
        lines.append(f"    [OK] {c}")

    lines.append("\n[6] УВЕРЕННОСТЬ")
    lines.append(f"    {rec.confidence:.2f}")

    lines.append("\n[7] ОБЪЯСНЕНИЕ")
    for chunk in _wrap(rec.explanation, w - 4):
        lines.append(f"    {chunk}")

    if rec.alternatives:
        lines.append("\n[8] ДОПУСТИМЫЕ АЛЬТЕРНАТИВЫ")
        for a in rec.alternatives:
            d = ", ".join(f"{k} {v:+.1f}" for k, v in a["deltas"].items())
            lines.append(
                f"    {a['candidate_id']}  {d:<40} "
                f"S_hi={a['sulfur_hi']}  cost={a['cost_proxy']}"
            )

    lines.append("=" * w)
    return "\n".join(lines)


def _wrap(text: str, width: int):
    words, line, out = text.split(), "", []
    for wd in words:
        if len(line) + len(wd) + 1 > width:
            out.append(line)
            line = wd
        else:
            line = f"{line} {wd}".strip()
    if line:
        out.append(line)
    return out
