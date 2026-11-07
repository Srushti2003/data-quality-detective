
from unittest.mock import MagicMock, call

import pandas as pd
import pytest

from src.evaluation import anomaly_evaluator as evaluator


def make_engine():
    engine = MagicMock()

    connection = MagicMock()
    connection_context = MagicMock()
    connection_context.__enter__.return_value = connection
    engine.connect.return_value = connection_context

    transaction = MagicMock()
    transaction_context = MagicMock()
    transaction_context.__enter__.return_value = transaction
    engine.begin.return_value = transaction_context

    return engine, connection, transaction


def make_trip_dataframe():
    return pd.DataFrame({
        "trip_id": [1, 2, 3, 4],
    })


def make_ground_truth():
    return pd.DataFrame({
        "trip_id": [2, 4],
    })


def make_predictions():
    return {
        "IQR": pd.DataFrame({"trip_id": [2, 3]}),
        "Z_SCORE": pd.DataFrame({"trip_id": [2]}),
        "MODIFIED_Z": pd.DataFrame({"trip_id": [4]}),
        "ISOLATION_FOREST": pd.DataFrame({"trip_id": [1, 4]}),
    }


def configure_main(monkeypatch, predictions=None):
    engine, connection, transaction = make_engine()

    monkeypatch.setattr(
        evaluator,
        "get_engine",
        MagicMock(return_value=engine),
    )

    monkeypatch.setattr(
        evaluator.pd,
        "read_sql",
        MagicMock(return_value=make_trip_dataframe()),
    )

    monkeypatch.setattr(
        evaluator,
        "load_ground_truth",
        MagicMock(return_value=make_ground_truth()),
    )

    prediction_data = predictions or make_predictions()

    def load_prediction(connection_arg, method):
        return prediction_data[method]

    monkeypatch.setattr(
        evaluator,
        "load_predictions",
        MagicMock(side_effect=load_prediction),
    )

    save_mock = MagicMock()
    monkeypatch.setattr(evaluator, "save_results", save_mock)

    return engine, connection, transaction, save_mock


def test_main_evaluates_all_detection_methods_and_saves_results(
    monkeypatch,
    capsys,
):
    engine, connection, transaction, save_mock = configure_main(
        monkeypatch
    )

    evaluator.main()

    assert evaluator.pd.read_sql.call_count == 1
    evaluator.load_ground_truth.assert_called_once_with(connection)

    assert evaluator.load_predictions.call_args_list == [
        call(connection, method)
        for method in evaluator.DETECTION_METHODS
    ]

    save_mock.assert_called_once()

    saved_transaction, results = save_mock.call_args.args

    assert saved_transaction is transaction
    assert len(results) == 4

    assert [
        result["detection_method"]
        for result in results
    ] == evaluator.DETECTION_METHODS

    assert engine.connect.called
    assert engine.begin.called
    engine.dispose.assert_called_once()

    output = capsys.readouterr().out

    assert "Loading evaluation data..." in output
    assert "ANOMALY DETECTION EVALUATION" in output
    assert "IQR" in output
    assert "Z_SCORE" in output
    assert "MODIFIED_Z" in output
    assert "ISOLATION_FOREST" in output
    assert "Evaluation saved to PostgreSQL." in output


def test_main_rejects_empty_trip_dataframe(monkeypatch):
    engine, connection, transaction, save_mock = configure_main(
        monkeypatch
    )

    monkeypatch.setattr(
        evaluator.pd,
        "read_sql",
        MagicMock(return_value=pd.DataFrame(columns=["trip_id"])),
    )

    with pytest.raises(ValueError, match="No records found"):
        evaluator.main()

    save_mock.assert_not_called()
    engine.dispose.assert_called_once()


def test_main_rejects_empty_ground_truth(monkeypatch):
    engine, connection, transaction, save_mock = configure_main(
        monkeypatch
    )

    monkeypatch.setattr(
        evaluator,
        "load_ground_truth",
        MagicMock(return_value=pd.DataFrame(columns=["trip_id"])),
    )

    with pytest.raises(
        ValueError,
        match="No ground-truth labels found",
    ):
        evaluator.main()

    evaluator.load_predictions.assert_not_called()
    save_mock.assert_not_called()
    engine.dispose.assert_called_once()


@pytest.mark.parametrize(
    "missing_method",
    evaluator.DETECTION_METHODS,
)
def test_main_rejects_missing_predictions(
    monkeypatch,
    missing_method,
):
    predictions = make_predictions()

    predictions[missing_method] = pd.DataFrame(
        columns=["trip_id"]
    )

    engine, connection, transaction, save_mock = configure_main(
        monkeypatch,
        predictions,
    )

    with pytest.raises(
        ValueError,
        match=f"No predictions found for {missing_method}",
    ):
        evaluator.main()

    save_mock.assert_not_called()
    engine.begin.assert_not_called()
    engine.dispose.assert_called_once()


def test_main_disposes_engine_when_initial_database_read_fails(
    monkeypatch,
):
    engine, connection, transaction, save_mock = configure_main(
        monkeypatch
    )

    monkeypatch.setattr(
        evaluator.pd,
        "read_sql",
        MagicMock(side_effect=RuntimeError("Database read failed")),
    )

    with pytest.raises(
        RuntimeError,
        match="Database read failed",
    ):
        evaluator.main()

    save_mock.assert_not_called()
    engine.dispose.assert_called_once()


def test_main_disposes_engine_when_prediction_loading_fails(
    monkeypatch,
):
    engine, connection, transaction, save_mock = configure_main(
        monkeypatch
    )

    monkeypatch.setattr(
        evaluator,
        "load_predictions",
        MagicMock(side_effect=RuntimeError("Prediction query failed")),
    )

    with pytest.raises(
        RuntimeError,
        match="Prediction query failed",
    ):
        evaluator.main()

    save_mock.assert_not_called()
    engine.dispose.assert_called_once()


def test_main_disposes_engine_when_saving_results_fails(
    monkeypatch,
):
    engine, connection, transaction, save_mock = configure_main(
        monkeypatch
    )

    save_mock.side_effect = RuntimeError("Database save failed")

    with pytest.raises(
        RuntimeError,
        match="Database save failed",
    ):
        evaluator.main()

    save_mock.assert_called_once()
    engine.dispose.assert_called_once()


def test_main_uses_configured_batch_id(monkeypatch):
    engine, connection, transaction, save_mock = configure_main(
        monkeypatch
    )

    monkeypatch.setattr(evaluator, "BATCH_ID", 99)

    evaluator.main()

    read_call = evaluator.pd.read_sql.call_args

    assert read_call.kwargs["params"] == {"batch_id": 99}

    evaluator.load_ground_truth.assert_called_once_with(connection)

    engine.dispose.assert_called_once()
