from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _pct(value: float) -> str:
    return f"{value:.2%}"


def write_final_report(
    output_path: Path,
    *,
    dataset_summary: dict[str, Any],
    cv_table: pd.DataFrame,
    selected_model: str,
    policy: dict[str, Any],
    test_metrics: dict[str, Any],
    bootstrap: dict[str, dict[str, float]],
    factual_metrics: dict[str, float],
    business: dict[str, Any],
) -> None:
    comparison = cv_table.to_markdown(index=False, floatfmt=".5f")
    if policy["fraction"] > 0:
        policy_text = (
            f"OOF-оценка рекомендует таргетировать верхние "
            f"{policy['fraction']:.1%} клиентов по score. Оценённый uplift в этой "
            f"группе — {policy['uplift_rate']:.2%}, ожидаемое число предотвращённых "
            f"оттоков — {policy['estimated_prevented_churns']:.1f}, условная "
            f"чистая ценность — {policy['estimated_net_value']:.1f}."
        )
    else:
        policy_text = (
            "При заданных стоимостных предположениях OOF-оценка не подтверждает "
            "положительную ценность кампании. Корректное решение — не запускать "
            "массовое воздействие без нового эксперимента или изменения экономики."
        )

    report = f"""# Финальный отчёт

## 1. Постановка задачи

Цель — ранжировать клиентов по ожидаемому снижению вероятности оттока вследствие
retention-воздействия. Это treatment-aware задача: высокий риск оттока сам по себе
не означает, что клиент изменит решение после предложения.

## 2. Данные

Используется реальный анонимизированный датасет Orange Belgium / ULB:
{dataset_summary['rows']:,} наблюдений, {dataset_summary['features']} признаков,
доля treatment {dataset_summary['treatment_rate']:.2%}, churn rate в контроле
{dataset_summary['control_churn_rate']:.2%}, в treatment
{dataset_summary['treated_churn_rate']:.2%}.

Кампании проводились с сентября по декабрь 2020 года; churn измерялся в
двухмесячном окне. Treatment был рандомизирован, поэтому данные позволяют
оценивать эффект вмешательства.

## 3. Валидация без leakage

- Один stratified holdout выделен до выбора модели.
- Выбор модели и размера кампании выполнен только по out-of-fold прогнозам.
- Holdout использован один раз для итоговой оценки.
- Stratification учитывает совместные классы `(treatment, churn)`.

В наборе нет идентификатора месяца кампании, поэтому честный temporal split
невозможен. Это ограничение нельзя скрывать или заменять выдуманной датой.

## 4. Сравнение моделей по OOF

{comparison}

Выбрана модель **{selected_model}** по максимальному OOF AUUC. Risk-only модель
оставлена как сильный benchmark: на малых uplift-выборках она иногда устойчивее
прямых causal learners.

## 5. Политика кампании

{policy_text}

Стоимостные предположения параметризованы, а не выдаются за известные факты:
ценность предотвращённого оттока {business['customer_value']}, стоимость контакта
{business['contact_cost']}, стоимость предложения {business['offer_cost']}.

## 6. Итоговая holdout-оценка

- AUUC: {test_metrics['auuc']:.5f}
- Qini: {test_metrics['qini']:.5f}
- Uplift@{business['top_k_fraction']:.0%}: {_pct(test_metrics['uplift_rate'])}
- ROC-AUC factual outcome: {factual_metrics['roc_auc']:.4f}
- Average precision: {factual_metrics['average_precision']:.4f}
- Brier score: {factual_metrics['brier_score']:.4f}

Bootstrap 95% CI:

- AUUC: [{bootstrap['auuc']['ci_low']:.5f}, {bootstrap['auuc']['ci_high']:.5f}]
- Qini: [{bootstrap['qini']['ci_low']:.5f}, {bootstrap['qini']['ci_high']:.5f}]
- Uplift@{business['top_k_fraction']:.0%}: [
  {_pct(bootstrap['uplift_at_k']['ci_low'])},
  {_pct(bootstrap['uplift_at_k']['ci_high'])}
  ]

Широкий интервал или интервал, включающий ноль, означает, что эффект нельзя
считать надёжно подтверждённым.

## 7. Интерпретация

Feature importance относится к анонимизированным PCA-компонентам и факторам.
Она показывает используемый моделью сигнал, но не позволяет назвать
человекочитаемые причины оттока и тем более не доказывает причинность.

## 8. Бизнес-рекомендации

1. Если новая экспериментальная проверка и экономика подтвердят запуск,
   использовать score как batch-ranking при фиксированном бюджете кампании.
2. Проверять policy на новой рандомизированной holdout-группе.
3. Пересчитывать экономику при изменении CLV, стоимости контакта и предложения.
4. Не таргетировать клиентов лишь по высокому churn risk: часть из них может быть
   `lost cause` или `do-not-disturb`.

## 9. Ограничения и улучшения

- признаки анонимизированы, поэтому невозможна содержательная диагностика причин;
- кампании относятся к 2020 году, возможен temporal/domain drift;
- в опубликованном наборе нет campaign timestamp;
- outcome редкий, поэтому uplift-оценки имеют высокую дисперсию;
- следующие шаги: новые временные данные, repeated cross-fitting, DR/X-learner,
  sensitivity analysis и prospective A/B test.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
